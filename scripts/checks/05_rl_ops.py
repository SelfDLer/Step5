"""Preflight Step5 kernels or inventory an actual A3 GRPO profiling run."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import xml.etree.ElementTree as ET


def absolute(value):
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("expected an absolute path")
    return path


def preflight(args):
    py = sys.executable
    checks = Path(__file__).resolve().parent
    fused = args.fsdp_src / "tests/step5_fused_ops"
    triton = args.ascend_src / "tests/e2e/nightly/single_node/ops/singlecard_ops/triton"
    cases = {
        "gbsa_gbsag": [checks / "03_check_gbsa.py", "--expected-prefix", sys.prefix],
        "indexer_i64": [checks / "04_check_indexer_rope.py", "--ascend-src", args.ascend_src, "--positions", "int64"],
        "indexer_i32": [checks / "04_check_indexer_rope.py", "--ascend-src", args.ascend_src, "--positions", "int32"],
        "fsdp_norm_rope_grad": ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
                                fused / "check_fsdp_norm_rope_npu.py"],
        "swiglu_grad": ["-m", "pytest", "-k", "npu", fused / "test_step5_swiglu.py"],
        "decode_sparse_meta": ["-m", "pytest", triton / "test_step5_decode_sparse_meta.py"],
        "fused_add_norm": ["-m", "pytest", triton / "test_step5_fused_add_rms_norm.py"],
    }
    if not args.ascend_src.is_dir() or not args.fsdp_src.is_dir():
        raise SystemExit("check --ascend-src and --fsdp-src")
    if args.device < 0:
        raise SystemExit("choose one nonnegative, idle NPU device ID")
    args.out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "ASCEND_RT_VISIBLE_DEVICES": str(args.device)}
    states = {}
    for name, command in cases.items():
        log = args.out / (name + ".log")
        junit = args.out / (name + ".xml")
        pytest_case = command[:2] == ["-m", "pytest"]
        if pytest_case:
            command += ["-q", "-rs", f"--junitxml={junit}"]
        print(f"RUN {name} -> {log}", flush=True)
        with log.open("w") as stream:
            rc = subprocess.run([py, *command], env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        state = "FAIL" if rc else "PASS"
        if pytest_case and not rc:
            try:
                tests = ET.parse(junit).findall(".//testcase")
            except (OSError, ET.ParseError):
                tests = []
            if not tests or any(t.find("skipped") is not None for t in tests):
                state = "INCOMPLETE"
        states[name] = state
        print(f"{state} {name}", flush=True)
    (args.out / "results.json").write_text(json.dumps(states, indent=2))
    return 0 if all(s == "PASS" for s in states.values()) else 1


def database_rows(path):
    # Each query aggregates one profiler event source before returning it to Python.
    queries = {
        "kernel": """SELECT CAST(t.deviceId AS TEXT), s.value, '', '', count(*)
                     FROM COMPUTE_TASK_INFO c JOIN TASK t USING(globalTaskId)
                     JOIN STRING_IDS s ON s.id=c.name GROUP BY 1,2""",
        "host": """SELECT '', n.value, coalesce(sh.value,''), coalesce(dt.value,''), count(*)
                   FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name
                   LEFT JOIN STRING_IDS sh ON sh.id=p.inputShapes
                   LEFT JOIN STRING_IDS dt ON dt.id=p.inputDtypes GROUP BY 2,3,4""",
        "hccl": """SELECT '', s.value, '', '', count(*) FROM COMMUNICATION_OP c
                   JOIN STRING_IDS s ON s.id=c.opName GROUP BY 2""",
    }
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for kind, table in (("kernel", "COMPUTE_TASK_INFO"), ("host", "PYTORCH_API"), ("hccl", "COMMUNICATION_OP")):
            if table in tables:
                for device, name, shape, dtype, n in db.execute(queries[kind]):
                    yield kind, device, name, shape, dtype, n


def rollout_rows(kernel):
    devices = set()
    with kernel.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            device = row["Device_id"].strip()
            devices.add(device)
            yield "kernel", device, row["Name"], row.get("Input Shapes", ""), row.get("Input Data Types", ""), 1
    device = next(iter(devices)) if len(devices) == 1 else ""
    host = kernel.with_name("operator_details.csv")
    if host.exists():
        with host.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                yield "host", device, row["Name"], row.get("Input Shapes", ""), "", 1
    comm = kernel.with_name("communication.json")
    if comm.exists():
        for step in json.loads(comm.read_text()).values():
            for group in ("p2p", "collective"):
                for name in step.get(group, {}):
                    if not name.lower().startswith("total"):
                        yield "hccl", device, name, "", "", 1


def inventory(args):
    run = args.run_dir
    if not run.is_dir():
        raise SystemExit("run directory missing")
    counts = Counter()
    devices = defaultdict(set)
    kinds = defaultdict(set)
    errors = []
    for role, pattern, parser in (("actor", "ascend_pytorch_profiler*.db", database_rows),
                                  ("rollout", "kernel_details.csv", rollout_rows)):
        for path in sorted((run / "profiles" / role).rglob(pattern)):
            try:
                for kind, device, name, shape, dtype, n in parser(path):
                    counts[role, kind, device, name, shape, dtype] += n
                    kinds[role].add(kind)
                    if kind == "kernel" and device != "":
                        devices[role].add(device)
            except (OSError, ValueError, KeyError, sqlite3.Error, csv.Error) as exc:
                errors.append(f"{path}: {exc}")
    with (run / "operators.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(("role", "source", "device", "name", "input_shapes", "input_dtypes", "record_count"))
        for key, count in sorted(counts.items()):
            writer.writerow((*key, count))
    step = False
    log = run / "train.log"
    if log.is_file():
        with log.open(errors="replace") as stream:
            step = any(re.search(r"\bstep:\s*1\b", line) and "training/global_step" in line for line in stream)
    if not step or not (run / "ops-trainer-exited-ok").is_file():
        errors.append("missing step 1 training metrics or successful trainer exit")
    for role in ("actor", "rollout"):
        print(role, "kernel devices:", sorted(devices[role]), "record sources:", sorted(kinds[role]))
        if len(devices[role]) != 16 or kinds[role] != {"kernel", "host", "hccl"}:
            errors.append(f"{role}: need 16 distinct kernel devices and host/kernel/HCCL records")
    for message in errors:
        print(message, file=sys.stderr)
    print("PARTIAL" if errors else "STEP_TRACED", run / "operators.csv")
    return int(bool(errors))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    one = modes.add_parser("preflight")
    one.add_argument("--ascend-src", type=absolute, required=True)
    one.add_argument("--fsdp-src", type=absolute, required=True)
    one.add_argument("--out", type=absolute, required=True)
    one.add_argument("--device", type=int, required=True)
    report = modes.add_parser("inventory")
    report.add_argument("--run-dir", type=absolute, required=True)
    args = cli.parse_args()
    return preflight(args) if args.mode == "preflight" else inventory(args)


if __name__ == "__main__":
    sys.exit(main())
