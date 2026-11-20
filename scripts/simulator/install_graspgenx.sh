#!/usr/bin/env bash
# Install the local eager inference service in the active simulation environment.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
config="$repo/configs/simulators"
revision=44eb1d77d6d27bc97ee361e3afb717a2115e30a9
vendor="${XDG_DATA_HOME:-$HOME/.local/share}/jiuwensymbiosis/vendor"
checkout="$vendor/GraspGenX"

if [[ ! -e "$checkout" ]]; then
    mkdir -p "$vendor"
    staging=$(mktemp -d "$vendor/.graspgenx-XXXXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    git init -q "$staging"
    git -C "$staging" remote add origin https://github.com/NVlabs/GraspGenX.git
    git -C "$staging" fetch --depth 1 origin "$revision"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$staging" checkout -q --detach FETCH_HEAD
    mv -T -- "$staging" "$checkout"
fi

if [[ ! -e "$checkout/.git" || "$(git -C "$checkout" rev-parse HEAD)" != "$revision" ]]; then
    echo "Expected GraspGenX commit $revision at $checkout. Check the existing directory." >&2
    exit 1
fi

patch="$config/graspgenx-runtime.patch"
if ! git -C "$checkout" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$checkout" apply "$patch"
fi

python -m pip install -c "$config/runtime-constraints-linux.txt" -e "$checkout[serve]"
python -m pip check
