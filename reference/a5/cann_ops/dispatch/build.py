"""Build the Step5 BF16 dispatch specialization against the activated Ascend 950 SDK."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

SOURCE = Path(__file__).resolve().parent
HEADER = Path("moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h")
KERNEL = Path("moe_distribute_dispatch_v3/op_kernel/arch35/moe_distribute_dispatch_v3_apt.cpp")
STEM = "MoeDistributeDispatchV3_e92e3b86d5d6432a166d18733d3c3613"


def build(output):
    opp = Path(os.environ["ASCEND_OPP_PATH"])
    sdk = opp / "built-in/op_impl/ai_core/tbe/impl/ops_transformer"
    output.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(dir=output) as temporary:
        work = Path(temporary)
        tree = work / "ascendc"
        for op in (HEADER.parts[0], KERNEL.parts[0]):
            shutil.copytree(sdk / "ascendc" / op, tree / op)
        (tree / "common").symlink_to(sdk / "ascendc/common", target_is_directory=True)
        shutil.copyfile(SOURCE / HEADER.name, tree / HEADER)
        shutil.copyfile(SOURCE / KERNEL.name, tree / KERNEL)

        # opc may relocate its Python entry. Bind source selection to this private build tree.
        entry = sdk / "dynamic/moe_distribute_dispatch_v3_apt.py"
        driver = entry.read_text()
        lookup = "src = get_kernel_source(ascendc_src_file, ascendc_src_dir, ascendc_src_dir_ex)"
        if driver.count(lookup) != 1:
            raise RuntimeError(f"Unsupported CANN dispatch compiler entry: {entry}")
        private_entry = work / entry.name
        private_entry.write_text(driver.replace(lookup, f"src = {str(tree / KERNEL)!r}"))
        subprocess.run([
            "opc", str(private_entry), "--main_func=moe_distribute_dispatch_v3",
            f"--input_param={SOURCE / 'input.json'}", "--soc_version=Ascend950PR_9599",
            f"--output={work}", "--impl_mode=high_performance,optional",
            "--simplified_key_mode=0", "--op_mode=dynamic",
            "--optional_input_mode=gen_placeholder", "--optional_output_mode=gen_placeholder",
        ], cwd=work, env={**os.environ, "ASCEND_CACHE_PATH": str(work / "cache"), "ASCEND_MAX_OPCACHE_SIZE": "0"}, check=True)
        artifacts = [work / (STEM + suffix) for suffix in (".o", ".json")]
        if not all(path.is_file() and path.stat().st_size for path in artifacts):
            raise RuntimeError("opc did not produce the expected Ascend 950 specialization")
        for path in artifacts:
            path.rename(output / path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory")
    build(parser.parse_args().output.resolve())
