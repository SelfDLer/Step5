"""Preflight Step5 kernels or inventory an actual A3 GRPO profiling run."""
import argparse
from collections import Counter, defaultdict
import csv
from importlib import metadata, util
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from step5_rollout_cases import ROLLOUT_CASES, prepare_rollout_case


def absolute(value):
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("expected an absolute path")
    return path


def compile_records(cache):
    """Map (Triton cache key, kernel name) to completion evidence on Ascend."""
    entries = defaultdict(set)
    for path in cache.rglob("*"):
        if path.is_file() and path.suffix in {".source", ".ttir", ".ttadapter", ".npubin"}:
            entries[path.parent, path.stem].add(path.suffix)
    records = {}
    for (directory, name), suffixes in entries.items():
        complete = (".npubin" in suffixes and (directory / f"{name}.json").is_file()
                    and (directory / f"__grp__{name}.json").is_file())
        key = (directory.name, name)
        records[key] = records.get(key, False) or complete
    return records


def write_states(out, states):
    pending = out / "results.json.tmp"
    pending.write_text(json.dumps(states, indent=2))
    pending.replace(out / "results.json")


def pytest_result(junit, rc):
    """Use pytest's call-phase failure records, including for mixed outcomes."""
    try:
        tests = ET.parse(junit).findall(".//testcase")
    except (OSError, ET.ParseError) as exc:
        return "INCOMPLETE", f"pytest exit {rc}, no usable JUnit report ({exc})"
    counts = Counter()
    for test in tests:
        if test.find("failure") is not None:
            counts["call failures"] += 1
        if test.find("skipped") is not None:
            counts["skipped"] += 1
        for error in test.findall("error"):
            message = error.get("message", "")
            phase = next((p for p in ("setup", "teardown") if message.startswith(f"failed on {p}")), None)
            if not phase:
                phase = "collection" if message == "collection failure" else "runner"
            counts[f"{phase} errors"] += 1
    evidence = f"pytest exit {rc}, JUnit testcase records={len(tests)}"
    for outcome, count in counts.items():
        evidence += f", {outcome}={count}"
    if counts["call failures"]:
        return "FAIL", evidence
    if rc or not tests or counts:
        return "INCOMPLETE", evidence
    return "PASS", evidence


def record_environment(args):
    packages = {}
    for dist in ("torch", "torch-npu", "triton", "triton-ascend", "vllm-ascend"):
        try:
            packages[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            packages[dist] = None
    spec = util.find_spec("triton")
    data = {"packages": packages, "python": sys.executable, "triton_file": spec.origin if spec else None,
            "ascend_home": os.getenv("ASCEND_HOME_PATH"), "device": args.device,
            "inherited_VLLM_USE_MODELSCOPE": os.getenv("VLLM_USE_MODELSCOPE"),
            "preflight_VLLM_USE_MODELSCOPE": "false"}
    for role, src in (("ascend", args.ascend_src), ("fsdp", args.fsdp_src)):
        rev = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True)
        data[role + "_src"] = str(src)
        data[role + "_commit"] = rev.stdout.strip()
    (args.out / "environment.json").write_text(json.dumps(data, indent=2))


def kernel_map(src, out, operators=None):
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        pending = Path(tmp) / out.name
        command = [sys.executable, Path(__file__).with_name("06_step5_kernel_map.py"),
                   "--ascend-src", src, "--out", pending]
        if operators:
            command += ["--operators", operators]
        rc = subprocess.run(command).returncode
        if not rc:
            pending.replace(out)
        return rc


def write_preflight_coverage(out, states):
    with (out / "kernel-map.csv").open(newline="", encoding="utf-8") as stream:
        sites = list(csv.DictReader(stream))
    by_name = defaultdict(set)
    passed = defaultdict(set)
    attempted = defaultdict(set)
    for site in sites:
        symbol = site["symbol"]
        by_name[symbol.split(".")[-1]].add(symbol)
    for case, state in states.items():
        for (_, kernel), complete in compile_records(out / "triton-cache" / case).items():
            attempted[kernel].add(case)
            if state == "PASS" and complete:
                passed[kernel].add(case)
    statuses = defaultdict(set)
    with (out / "preflight-kernels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("symbol", "source_use", "launch_site", "status", "cases"))
        for site in sites:
            symbol = site["symbol"]
            kernel = symbol.split(".")[-1]
            use = site["source_use"]
            if use != "launched":
                status = use.upper()
            elif kernel not in attempted:
                status = "NO_PREFLIGHT_EVIDENCE"
            elif len(by_name[kernel]) > 1:
                status = "AMBIGUOUS_NAME"
            elif kernel not in passed:
                status = "NO_PASSING_CASE"
            else:
                status = "COMPILED_IN_PASSING_CASE"
            statuses[status].add(symbol)
            case_names = passed[kernel] if status == "COMPILED_IN_PASSING_CASE" else attempted.get(kernel, ())
            writer.writerow((symbol, use, site["launch_site"], status, ",".join(sorted(case_names))))
    print("Static kernel coverage (distinct symbols):", {s: len(v) for s, v in sorted(statuses.items())})


def preflight(args):
    py = sys.executable
    checks = Path(__file__).resolve().parent
    fused = args.fsdp_src / "tests/step5_fused_ops"
    ascend_ops = args.ascend_src / "tests/e2e/nightly/single_node/ops"
    triton = ascend_ops / "singlecard_ops/triton"
    cases = {
        "gbsa_gbsag": [checks / "03_check_gbsa.py", "--expected-prefix", sys.prefix, "--kv-heads", "4",
                       "--lengths", "7", "8", "9", "127", "128", "129"],
        "indexer_i64": [checks / "04_check_indexer_rope.py", "--ascend-src", args.ascend_src, "--positions", "int64"],
        "indexer_i32": [checks / "04_check_indexer_rope.py", "--ascend-src", args.ascend_src, "--positions", "int32"],
        "fsdp_norm_rope_grad": ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
                                fused / "check_fsdp_norm_rope_npu.py"],
        "swiglu_grad": ["-m", "pytest", "-k", "npu", fused / "test_step5_swiglu.py"],
        "grouped_matmul_grad": ["-m", "pytest", "-k", "npu", str(args.fsdp_src /
            "tests/unit_tests/ops/test_grouped_matmul_gradients.py") + "::test_grouped_matmul_requested_gradients"],
        "dense_tnd_grad": ["-m", "pytest", str(args.fsdp_src / "tests/unit_tests/ops/test_step5_attention_tnd.py")
                           + "::test_real_npu_dense_tnd_forward_backward"],
        "gbsa_padded_batch_grad": ["-m", "pytest", str(args.fsdp_src / "tests/unit_tests/models/test_step5.py")
                                   + "::test_step5_gbsa_batched_right_padding_matches_single_samples"],
        "gbsa_packed_tnd_grad": ["-m", "pytest", "-k", "full", str(args.fsdp_src /
            "tests/unit_tests/ops/test_gbsa_tnd.py") + "::test_npu_step5_tnd_matches_bsnd_gbsa"],
        "moe_route_grad": ["-m", "pytest", checks / "test_step5_routing_npu.py"],
        "rollout_moe_gate": ["-m", "pytest", checks / "test_rollout_moe_gate_npu.py"],
        "rollout_swiglustep": ["-m", "pytest", triton / "test_swiglustep.py"],
        "decode_sparse_meta": ["-m", "pytest", triton / "test_step5_decode_sparse_meta.py"],
        "csa_decode_flush": ["-m", "pytest", args.out / "test_step5_csa_decode_flush.py"],
        "fused_add_norm": ["-m", "pytest", triton / "test_step5_fused_add_rms_norm.py"],
    }
    for heads in (1, 2):
        cases[f"gbsa_gbsag_kv{heads}"] = [checks / "03_check_gbsa.py", "--expected-prefix", sys.prefix,
                                         "--kv-heads", str(heads), "--lengths", "7", "8", "9", "127", "128", "129"]
    cases.update(dict.fromkeys(ROLLOUT_CASES))
    for name in ("indexer_i64", "indexer_i32"):
        if args.model_config:
            cases[name] += ["--config", args.model_config]
        cases[name] += ["--tp", str(args.tp)]
    if not args.ascend_src.is_dir() or not args.fsdp_src.is_dir():
        raise SystemExit("check --ascend-src and --fsdp-src")
    if args.device < 0:
        raise SystemExit("choose one nonnegative, idle NPU device ID")
    if args.only:
        if args.only not in cases:
            raise SystemExit("unknown case; choose from " + ", ".join(cases))
        cases = {args.only: cases[args.only]}
    args.out.mkdir(parents=True, exist_ok=False)
    record_environment(args)
    map_status = 0
    if not args.only:
        map_status = kernel_map(args.ascend_src, args.out / "kernel-map.csv")
    env = {**os.environ, "ASCEND_RT_VISIBLE_DEVICES": str(args.device), "VLLM_USE_MODELSCOPE": "false",
           "STEP5_CHECK_FSDP_SRC": str(args.fsdp_src), "STEP5_CHECK_ASCEND_SRC": str(args.ascend_src)}
    states = dict.fromkeys(cases, "NOT_RUN")
    write_states(args.out, states)
    for name, command in cases.items():
        case_cache = args.out / "triton-cache" / name
        case_cache.mkdir(parents=True)
        case_env = {**env, "TRITON_CACHE_DIR": str(case_cache), "TRITON_STORE_BINARY_ONLY": "0"}
        if name in ROLLOUT_CASES:
            try:
                command = prepare_rollout_case(name, args.ascend_src, args.out)
            except (ValueError, OSError, SyntaxError) as exc:
                states[name] = "INCOMPLETE"
                write_states(args.out, states)
                (args.out / (name + ".log")).write_text(str(exc))
                print(f"INCOMPLETE {name}: {exc}", flush=True)
                continue
        if name == "csa_decode_flush":
            # Keep the pinned numerical and slot-release assertions; the upstream
            # test still imports the step4 module although rollout invokes step5.
            original = triton / "test_step4_csa_decode_flush.py"
            old_import = "from vllm_ascend.models.step4.summary_kernels import csa_compact_decode_update"
            new_import = old_import.replace(".step4.", ".step5.")
            src = original.read_text() if original.is_file() else ""
            if src.count(old_import) != 1:
                states[name] = "INCOMPLETE"
                write_states(args.out, states)
                print(f"INCOMPLETE {name}: reference test changed or missing at {original}", flush=True)
                continue
            command[-1].write_text(src.replace(old_import, new_import))
            case_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(args.ascend_src), env.get("PYTHONPATH")]))
        log = args.out / (name + ".log")
        junit = args.out / (name + ".xml")
        pytest_case = command[:2] == ["-m", "pytest"]
        if pytest_case:
            # The E2E parent conftest imports ModelScope for unrelated LoRA fixtures.
            # Keep the ops conftest: it initializes Triton and registers NPU operations.
            test_files = [Path(str(arg).partition("::")[0]) for arg in command[2:]]
            parents = {path.parent.resolve() for path in test_files if path.is_absolute() and path.suffix == ".py"}
            if parents == {args.out.resolve()}:
                command += [f"--confcutdir={args.out}"]
            elif parents == {checks}:
                command += [f"--confcutdir={checks}"]
            elif parents and all(path.is_relative_to(ascend_ops.resolve()) for path in parents):
                command += [f"--confcutdir={ascend_ops}"]
            command += ["-q", "-rs", "--maxfail=0", f"--junitxml={junit}"]
        print(f"RUN {name} -> {log}", flush=True)
        with log.open("w") as stream:
            rc = subprocess.run([py, *command], env=case_env, stdout=stream, stderr=subprocess.STDOUT).returncode
        state = "FAIL" if rc else "PASS"
        if pytest_case:
            state, evidence = pytest_result(junit, rc)
            with log.open("a") as stream:
                print(f"\n[preflight] {evidence}", file=stream)
            print(evidence, flush=True)
        if state == "PASS" and name in {"indexer_i32", "indexer_i64", "csa_decode_flush"}:
            expected = "_csa_decode_stage_flush_kernel" if name == "csa_decode_flush" else "_indexer_qkz_norm_rope_kernel"
            if not any(kernel == expected and built for (_, kernel), built in compile_records(case_cache).items()):
                state = "INCOMPLETE"
                print(f"No completed cold compilation evidence for {expected} in {case_cache}", flush=True)
        states[name] = state
        write_states(args.out, states)
        print(f"{state} {name}", flush=True)
    if not args.only and not map_status:
        write_preflight_coverage(args.out, states)
    print("Test group results:", dict(Counter(states.values())), "->", args.out / "results.json")
    return int(bool(map_status) or any(s != "PASS" for s in states.values()))


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


def compare_compilations(run, preflight):
    if not (preflight / "results.json").is_file():
        print(f"Missing preflight results: {preflight}", file=sys.stderr)
        return False
    states = json.loads((preflight / "results.json").read_text())
    passed = defaultdict(set)
    names = defaultdict(set)
    for case, state in states.items():
        if state == "PASS":
            for key, complete in compile_records(preflight / "triton-cache" / case).items():
                if complete:
                    passed[key].add(case)
                    names[key[1]].add(case)
    runtime = compile_records(run / "triton-cache")
    totals = Counter()
    path = run / "triton-coverage.csv"
    with path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("kernel", "cache_key", "runtime_build", "preflight_match", "preflight_cases"))
        for key, complete in sorted(runtime.items(), key=lambda row: (row[0][1], row[0][0])):
            digest, kernel = key
            match = "EXACT_KEY" if key in passed else "NAME_ONLY" if kernel in names else "UNTESTED"
            build = "COMPLETE" if complete else "PARTIAL_ARTIFACTS"
            cases = passed.get(key, names.get(kernel, set()))
            writer.writerow((kernel, digest, build, match, ",".join(sorted(cases))))
            totals[build, match] += 1
    for (build, match), count in sorted(totals.items()):
        print(f"Triton {build} {match}: {count}")
    if not runtime:
        print("NO_CACHE_EVIDENCE: run has no recorded Triton builds", file=sys.stderr)
    print("Triton specialization report:", path)
    return bool(runtime) and totals["COMPLETE", "EXACT_KEY"] == len(runtime)


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
    if args.ascend_src and kernel_map(args.ascend_src, run / "kernel-map-vs-profile.csv", run / "operators.csv"):
        errors.append("failed to match source kernels to observed profiler records")
    if args.preflight_dir and not compare_compilations(run, args.preflight_dir):
        errors.append("unmatched or missing Triton compilation evidence")
    return int(bool(errors))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    one = modes.add_parser("preflight")
    one.add_argument("--ascend-src", type=absolute, required=True)
    one.add_argument("--fsdp-src", type=absolute, required=True)
    one.add_argument("--out", type=absolute, required=True)
    one.add_argument("--device", type=int, required=True)
    one.add_argument("--model-config", type=absolute, help="use this config.json for indexer layout and head counts")
    one.add_argument("--tp", type=int, default=8, help="rollout TP used to derive indexer geometry")
    one.add_argument("--only", help="run just one case, e.g. csa_decode_flush")
    report = modes.add_parser("inventory")
    report.add_argument("--run-dir", type=absolute, required=True)
    report.add_argument("--ascend-src", type=absolute, help="match rollout profile names against the pinned sources")
    report.add_argument("--preflight-dir", type=absolute, help="compare observed Triton compilation keys to successful preflight cases")
    args = cli.parse_args()
    return preflight(args) if args.mode == "preflight" else inventory(args)


if __name__ == "__main__":
    sys.exit(main())
