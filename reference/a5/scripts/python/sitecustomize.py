"""Select this checkout after the shared runtime's .pth hook has run."""
from pathlib import Path
import sys

import _step5_post7_shared_sources as wiring

src = Path(__file__).resolve().parents[2] / "src"
roots = [str(src / name) for name in ("FSDPTurbo", "verl", "vllm", "vllm-ascend")]
sys.path[:] = roots + [entry for entry in sys.path if entry not in roots]
wiring._SHARED_UTILS = str(src / "vllm-ascend/vllm_ascend/utils.py")
wiring._SHARED_VLLM_ENVS = str(src / "vllm/vllm/envs.py")

# A clean source checkout obtains the generated wheel version from the pinned runtime.
if not (src / "vllm/vllm/_version.py").is_file():
    wiring._NATIVE["vllm._version"] = str(Path(wiring.SHARED_SRC) / "vllm/vllm/_version.py")
