#!/usr/bin/env bash
# Run as libddla-runner after the administrator provisions the directories.
# Register without starting the service. Feed the short-lived token on stdin.
set -euo pipefail
[[ "$(id -un)" == libddla-runner ]]
ddla_root=/home/libddla-runner/ci
ddla_version=2.338.2
ddla_archive=$ddla_root/tmp/actions-runner-linux-x64-$ddla_version.tar.gz
umask 077
if [[ -e "$ddla_root/runner/.runner" ]]; then
    echo 'Runner is already registered; refusing to replace it.' >&2
    exit 1
fi
if [[ ! -x "$ddla_root/runner/bin/Runner.Listener" ]]; then
    curl --fail --location --retry 3 \
        "https://github.com/actions/runner/releases/download/v$ddla_version/actions-runner-linux-x64-$ddla_version.tar.gz" \
        --output "$ddla_archive"
    printf '%s  %s\n' baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40 "$ddla_archive" | sha256sum -c -
    tar -xzf "$ddla_archive" -C "$ddla_root/runner"
fi
IFS= read -r ddla_registration_token
[[ -n "$ddla_registration_token" ]]
(
    cd "$ddla_root/runner"
    ./config.sh --unattended --url https://github.com/userchba/LibDDLA \
        --token "$ddla_registration_token" --name sai-libddla-vm \
        --labels sai-libddla --work "$ddla_root/work"
)
unset ddla_registration_token
printf '%s\n' /usr/local/bin:/usr/bin:/bin > "$ddla_root/runner/.path"
echo 'Registered; the administrator can now verify and start libddla-runner.service.'
