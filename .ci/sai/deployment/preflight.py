#!/usr/bin/env python3
import grp
import os
from pathlib import Path
import pwd

if pwd.getpwuid(os.getuid()).pw_name != 'libddla-runner':
    raise SystemExit('Run only under the dedicated libddla-runner account')
groups = {grp.getgrgid(gid).gr_name for gid in os.getgroups()}
if groups & {'sudo', 'docker', 'lxd', 'disk', 'libvirt'}:
    raise SystemExit('Dedicated runner has a privileged supplementary group')
values = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
if values['MemAvailable'] < 3 * 1024 * 1024:
    raise SystemExit('Less than 3 GiB available; delaying runner startup')
if not Path('/home/libddla-runner/ci/ssh/config').is_file():
    raise SystemExit('Dedicated SSH configuration is missing')
