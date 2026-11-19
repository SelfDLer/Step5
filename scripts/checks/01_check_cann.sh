#!/usr/bin/env bash
# Run with bash after manually sourcing the intended CANN set_env.sh.
set -euo pipefail

printf '== Host ==\n'
uname -m
command -v python3 >/dev/null && python3 --version
command -v npu-smi >/dev/null || { echo 'FAIL: npu-smi is unavailable' >&2; exit 1; }
npu-smi info
npu-smi info -l
if [[ -n ${NPU_ID:-} ]]; then
    npu-smi info -t board -i "$NPU_ID"
else
    echo 'Set NPU_ID to an actual ID from the list and rerun to query driver/firmware details.'
fi

printf '\n== Activated CANN paths ==\n'
for key in CANN_ENV_SCRIPT ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_CUSTOM_OPP_PATH; do
    printf '%s=%s\n' "$key" "${!key-}"
done
: "${ASCEND_HOME_PATH:?Source the intended CANN set_env.sh first}"
: "${ASCEND_OPP_PATH:?CANN activation did not set ASCEND_OPP_PATH}"
[[ -d $ASCEND_HOME_PATH && -d $ASCEND_OPP_PATH ]] || { echo 'FAIL: CANN/OPP directory missing' >&2; exit 1; }
printf 'Resolved Toolkit: %s\n' "$(readlink -f -- "$ASCEND_HOME_PATH")"
printf 'Resolved OPP: %s\n' "$(readlink -f -- "$ASCEND_OPP_PATH")"

printf '\n== Package versions ==\n'
query_found=0
for root in "$ASCEND_HOME_PATH" "$(dirname -- "$ASCEND_HOME_PATH")"; do
    if [[ -f $root/query_pkg_version.sh ]]; then
        bash "$root/query_pkg_version.sh"
        query_found=1
        break
    fi
done
version_found=0
for file in \
    "$ASCEND_HOME_PATH/version.cfg" \
    "$ASCEND_HOME_PATH/version.info" \
    "$ASCEND_HOME_PATH/$(uname -m)-linux/ascend_toolkit_install.info" \
    "$ASCEND_OPP_PATH/version.info"; do
    if [[ -f $file ]]; then
        printf '\n--- %s ---\n' "$file"
        cat -- "$file"
        version_found=1
    fi
done
if [[ $query_found == 0 && $version_found == 0 ]]; then
    echo 'UNRESOLVED: no version record found; obtain exact Toolkit/ops versions from the provider.'
fi

printf '\n== Custom OPP version records, if supplied ==\n'
IFS=: read -r -a vendors <<< "${ASCEND_CUSTOM_OPP_PATH:-}"
for vendor in "${vendors[@]}"; do
    [[ -n $vendor ]] || continue
    [[ -d $vendor ]] || { echo "FAIL: missing custom OPP directory: $vendor" >&2; exit 1; }
    printf 'Vendor: %s\n' "$vendor"
    for file in "$vendor/version.info" "$vendor/version.cfg" "$vendor/manifest.json"; do
        if [[ -f $file ]]; then printf '\n--- %s ---\n' "$file"; cat -- "$file"; fi
    done
done
printf '\nINVENTORY COMPLETE: review versions and paths; this does not certify compatibility or GBSA support.\n'
