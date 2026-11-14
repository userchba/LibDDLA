#!/usr/bin/env python3
"""Run the dedicated VM worker with Landlock filesystem and TCP restrictions (no root)."""
import ctypes as c
import os
from pathlib import Path
import sys


class Ruleset(c.Structure):
    _fields_ = [("fs", c.c_uint64), ("net", c.c_uint64)]


class PathRule(c.Structure):
    _pack_ = 1
    _fields_ = [("access", c.c_uint64), ("parent", c.c_int)]


class PortRule(c.Structure):
    _fields_ = [("access", c.c_uint64), ("port", c.c_uint64)]


def restrict(root):
    libc = c.CDLL(None, use_errno=True)

    def checked(result):
        if result < 0:
            error = c.get_errno()
            raise OSError(error, os.strerror(error))
        return result

    abi = checked(libc.syscall(444, 0, 0, 1))
    if abi < 4:
        raise RuntimeError("Landlock ABI 4 or newer is required; refusing an unrestricted runner")
    fs = (1 << 15) - 1  # Execute/read/write/create/remove/refer/truncate.
    read = (1 << 0) | (1 << 2) | (1 << 3)
    attr = Ruleset(fs, 3)  # Handle TCP bind and connect.
    descriptor = checked(libc.syscall(444, c.byref(attr), c.sizeof(attr), 0))
    # System runtime is read-only; personal SSH files and other home directories
    # are absent from the allowlist. /proc is needed by the .NET runner.
    # Runner startup enumerates every parent directory. Directory names may be
    # listed, but opening file contents still requires a separate read grant.
    for name, access in [("/", 1 << 3), ("/usr", read), ("/bin", read), ("/sbin", read),
                         ("/lib", read), ("/lib64", read), ("/etc", read),
                         (str(Path("/etc/resolv.conf").resolve()), 1 << 2),
                         ("/proc", read), ("/sys", read), ("/dev", read | 2),
                         (str(root / "runner"), fs), (str(root / "work"), fs),
                         (str(root / "tmp"), fs), (str(root / "cache"), fs),
                         (str(root / "state"), fs), (str(root / "bin"), read),
                         (str(root / "ssh"), read),
                         (str(root / "deployment"), read)]:
        if not Path(name).exists():
            continue
        fd = os.open(name, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = PathRule(access, fd)
            checked(libc.syscall(445, descriptor, 1, c.byref(rule), 0))
        finally:
            os.close(fd)
    # GitHub HTTPS and the SAI SSH endpoint; disallow TCP listeners.
    for port in (443, 12022):
        rule = PortRule(2, port)
        checked(libc.syscall(445, descriptor, 2, c.byref(rule), 0))
    checked(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
    checked(libc.syscall(446, descriptor, 0))
    os.close(descriptor)


if __name__ == "__main__":
    if os.uname().machine != "x86_64":
        raise RuntimeError("This syscall mapping requires Linux x86_64")
    root = Path(sys.argv[1]).resolve()
    restrict(root)
    os.execvp(sys.argv[2], sys.argv[2:])
