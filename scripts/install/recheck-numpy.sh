#!/usr/bin/env bash
# Repair the numpy==2.5.5 conflict in the a5toa3 Python 3.12 environment.
set -euo pipefail
cd "$(dirname "$0")/../.."
prefix=/mnt/share/z00988734/.conda/step5
pins=.local/a3-install/verified-base.constraints.txt
python - "$prefix" "$pins" <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path
prefix, pins = map(Path, sys.argv[1:])
if Path(sys.prefix).resolve() != prefix.resolve() or sys.version_info[:2] != (3, 12):
    sys.exit(f'Activate the verified Python 3.12 environment at {prefix} first')
records = dict(line.split('==', 1) for line in pins.read_text().splitlines())
for name in ('torch', 'torch-npu'):
    if version(name) != records[name]:
        sys.exit(f'{name} differs from the verified version {records[name]}')
if records['numpy'] not in ('2.5.5', '1.26.4') or version('numpy') not in ('2.5.5', '1.26.4'):
    sys.exit('This repair applies to the recorded numpy 2.5.5 -> 1.26.4 migration')
PY
if grep -qx 'numpy==2.5.5' "$pins"; then
    cp -n -- "$pins" "${pins%.txt}.before-numpy-1.26.4.txt"
fi
python -m pip install --only-binary=:all: --no-deps 'numpy==1.26.4'
python -c 'import numpy; print(numpy.__version__, numpy.__file__); assert numpy.__version__ == "1.26.4"'
python scripts/checks/02_check_torch_npu.py --expected-prefix "$prefix" --device 0
python scripts/checks/03_check_gbsa.py --expected-prefix "$prefix" --device 0 --lengths 16 129 --kv-heads 4
new_pins=$(mktemp "${pins}.XXXXXX")
sed 's/^numpy==2\.5\.5$/numpy==1.26.4/' "$pins" > "$new_pins"
mv -- "$new_pins" "$pins"
echo 'NumPy 1.26.4 verified and constraints updated. Resume the deps stage.'
