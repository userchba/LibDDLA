#!/usr/bin/env bash
set -euo pipefail

root=${1:?rootfs destination is required}
rm -rf "$root"
mkdir -p "$root"/{bin,etc,dev,proc,sys,tmp,var/tmp,home,root,lib64}
mkdir -p \
    "$root/dev/infiniband" \
    "$root/run/nvidia-mps" \
    "$root/run/nvidia-mps-log" \
    "$root/usr/bin" \
    "$root/usr/share/glvnd/egl_vendor.d" \
    "$root/usr/share/vulkan/icd.d" \
    "$root/usr/share/vulkan/implicit_layer.d" \
    "$root/usr/share/egl/egl_external_platform.d" \
    "$root/usr/share/nvidia"

for path in \
    /usr/bin/nvidia-smi \
    /usr/bin/nvidia-debugdump \
    /usr/bin/nvidia-persistenced \
    /usr/bin/nvidia-cuda-mps-control \
    /usr/bin/nvidia-cuda-mps-server \
    /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    /usr/share/vulkan/icd.d/nvidia_icd.json \
    /usr/share/vulkan/icd.d/nvidia_layers.json \
    /usr/share/vulkan/implicit_layer.d/nvidia_layers.json \
    /usr/share/egl/egl_external_platform.d/15_nvidia_gbm.json \
    /usr/share/egl/egl_external_platform.d/10_nvidia_wayland.json \
    /usr/share/nvidia/nvoptix.bin
do
    : > "$root$path"
done

cp -L /bin/bash "$root/bin/bash"
ln -s bash "$root/bin/sh"
while read -r library; do
    mkdir -p "$root$(dirname "$library")"
    cp -L "$library" "$root$library"
done < <(ldd /bin/bash | sed -n 's/.*=> \([^ ]*\).*/\1/p')
cp -L /lib64/ld-linux-x86-64.so.2 "$root/lib64/ld-linux-x86-64.so.2"

cp -L /etc/hosts /etc/resolv.conf "$root/etc/"
printf 'root:x:0:0:root:/root:/bin/sh\n' > "$root/etc/passwd"
printf 'root:x:0:\n' > "$root/etc/group"
chmod 1777 "$root/tmp" "$root/var/tmp"
chmod -R a+rX "$root"
du -sh "$root"
