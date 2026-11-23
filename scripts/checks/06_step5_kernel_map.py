"""List Step5 rollout's source-level Triton launch sites without importing the model."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from pathlib import Path
import subprocess


PACKAGE = "vllm_ascend.models.step5"


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + "." + node.attr
    return ""


def bindings(tree, module):
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            prefix = node.module or ""
            if node.level:
                base = module.split(".")[:-node.level]
                prefix = ".".join(base + ([prefix] if prefix else []))
            for alias in node.names:
                result[alias.asname or alias.name] = prefix + "." + alias.name
        elif isinstance(node, ast.Import):
            result.update({a.asname: a.name for a in node.names if a.asname})
    return result


def qualified(name, aliases, scope, module):
    first, dot, rest = name.partition(".")
    if first == "self" and "." in scope:
        return module + "." + scope.rsplit(".", 1)[0] + dot + rest
    if first in aliases:
        return aliases[first] + dot + rest
    if dot:
        return name
    return module + "." + name


class Calls(ast.NodeVisitor):
    """Collect static edges including a launch's grid and specialization arguments."""
    def __init__(self, tree, module, path, definitions, edges, launches):
        self.module, self.path = module, path
        self.aliases = bindings(tree, module)
        self.scope = []
        self.definitions, self.edges, self.launches = definitions, edges, launches
        self.custom = {}
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
            if dotted(call.func).split(".")[-1] == "direct_register_custom_op":
                kw = {k.arg: k.value for k in call.keywords}
                if isinstance(kw.get("op_name"), ast.Constant) and isinstance(kw.get("op_func"), ast.Name):
                    self.custom["torch.ops.vllm." + kw["op_name"].value] = module + "." + kw["op_func"].id
        self.visit(tree)

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        key = ".".join([self.module, *self.scope])
        decorators = [d.func if isinstance(d, ast.Call) else d for d in node.decorator_list]
        jit = any(qualified(dotted(d), self.aliases, "", self.module) == "triton.jit" for d in decorators)
        self.definitions[key] = (self.path, node, jit)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        scope = ".".join(self.scope)
        caller = ".".join([self.module, *self.scope])
        launch = isinstance(node.func, ast.Subscript)
        name = dotted(node.func.value if launch else node.func)
        target = self.custom.get(name) or qualified(name, self.aliases, scope, self.module)
        self.edges[target].add((caller, f"{self.path}:{node.lineno}"))
        if launch:
            self.launches[target].append((caller, f"{self.path}:{node.lineno}", ast.unparse(node.func.slice),
                                          " ".join(ast.unparse(k) for k in node.keywords)))
        self.generic_visit(node)


def callers_of(target, edges, definitions):
    found = set()
    queue = [target]
    sites = set()
    while queue:
        for caller, site in edges[queue.pop()]:
            sites.add((caller, site))
            if caller in definitions and caller not in found:
                found.add(caller)
                queue.append(caller)
    return sorted(sites)


def abs_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("supply an absolute path")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascend-src", type=abs_path, required=True)
    parser.add_argument("--out", type=abs_path, required=True, help="new kernel map CSV")
    parser.add_argument("--operators", type=abs_path, help="optional operators.csv from 05_rl_ops.py inventory")
    args = parser.parse_args()
    root = args.ascend_src
    package = root.joinpath(*PACKAGE.split("."))
    if not (package / "model.py").is_file():
        parser.error(f"cannot find Step5 model sources in {package}")
    definitions, edges, launches = {}, defaultdict(set), defaultdict(list)
    for path in sorted(package.rglob("*.py")):
        rel = path.relative_to(root)
        Calls(ast.parse(path.read_text(encoding="utf-8")), ".".join(rel.with_suffix("").parts),
              rel.as_posix(), definitions, edges, launches)
    for path in sorted((root / "tests").rglob("test*.py")):
        text = path.read_text(encoding="utf-8")
        if PACKAGE not in text:
            continue
        rel = path.relative_to(root)
        Calls(ast.parse(text), ".".join(rel.with_suffix("").parts), rel.as_posix(), definitions, edges, launches)

    recorded = Counter()
    if args.operators:
        with args.operators.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream)
            if not {"role", "source", "name", "record_count"} <= set(rows.fieldnames or []):
                parser.error("unexpected operators.csv header")
            for row in rows:
                if row["role"] == "rollout" and row["source"] == "kernel":
                    recorded[row["name"]] += int(row["record_count"])
    jits = {key: val for key, val in definitions.items() if val[2] and key.startswith(PACKAGE + ".")}
    names = Counter(node.name for _, node, _ in jits.values())
    sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    states = Counter()
    with args.out.open("x", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("commit", "symbol", "definition", "source_use", "launch_site", "grid", "launch_kwargs",
                         "constexpr_parameters", "static_callers", "test_call_sites", "profile_name_status",
                         "matching_device_records"))
        for key, (path, node, _) in sorted(jits.items()):
            callers = callers_of(key, edges, definitions)
            direct = launches[key]
            source_use = "launched" if direct else "called_helper" if edges[key] else "unresolved"
            fields = [f"{fn}@{site}" for fn, site in callers if fn.startswith(PACKAGE + ".")]
            tests = [site for fn, site in callers if fn.startswith("tests.")]
            params = [a.arg for a in node.args.args if a.annotation and "constexpr" in ast.unparse(a.annotation)]
            records = recorded[node.name]
            if not args.operators:
                state = "no_profile_supplied"
            elif not records:
                state = "no_exact_name_match"
            else:
                state = "unique_name_seen" if names[node.name] == 1 else "ambiguous_name_seen"
            states[(source_use, state)] += 1
            for wrapper, location, grid, kwargs in direct or [("", "", "", "")]:
                writer.writerow((sha, key, f"{path}:{node.lineno}", source_use,
                                 f"{wrapper}@{location}" if wrapper else "", grid, kwargs,
                                 ";".join(params), ";".join(fields), ";".join(sorted(set(tests))), state, records))
    print(f"{len(jits)} JIT symbols in {package}; one row per static launch, or one per helper: {args.out}")
    for (use, state), count in sorted(states.items()):
        print(use, state, count)
    print("Static calls overapproximate branches. Dynamic dispatch and external/library kernels are outside this map.")
    print("Device names may be renamed or fused. A name match does not verify input specializations or correctness.")


if __name__ == "__main__":
    main()
