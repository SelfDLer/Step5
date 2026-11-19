SHARED_SRC = '/mnt/share/z00988734/projects/step5-re/src'
PRIVATE_SRC = '/mnt/share/z00988734/environments/step5-cann920-post7-rl-521/project/src'
_NATIVE = {'vllm._rust_tool_parser': '/mnt/share/z00988734/environments/step5-cann920-post7-rl-521/project/src/vllm/vllm/_rust_tool_parser.abi3.so', 'vllm_ascend.vllm_ascend_C': '/mnt/share/z00988734/environments/step5-cann920-post7-rl-521/project/src/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so'}
_PRIVATE_ASCEND = '/mnt/share/z00988734/environments/step5-cann920-post7-rl-521/project/src/vllm-ascend/vllm_ascend'
_RUST_EXECUTABLE = '/mnt/share/z00988734/environments/step5-cann920-post7-rl-521/project/src/vllm/vllm/vllm-rs'
_SHARED_UTILS = '/mnt/share/z00988734/projects/step5-re/src/vllm-ascend/vllm_ascend/utils.py'
_SHARED_VLLM_ENVS = '/mnt/share/z00988734/projects/step5-re/src/vllm/vllm/envs.py'

import importlib.machinery
import importlib.util
import os
import sys


class _AscendUtils(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        super().exec_module(module)
        if not hasattr(module, "_CUSTOM_OP_BASE_DIR"):
            raise ImportError("vllm_ascend.utils no longer provides its OPP base directory")
        module._CUSTOM_OP_BASE_DIR = _PRIVATE_ASCEND


class _RustEnv(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        super().exec_module(module)
        original = module._resolve_rust_cli_path

        def resolve():
            enabled = bool(int(os.getenv("VLLM_USE_RUST_FRONTEND", "0"))) or bool(int(os.getenv("VLLM_USE_RUST_BENCH", "0")))
            if enabled and os.getenv("VLLM_RUST_FRONTEND_PATH", "auto") == "auto":
                if not os.access(_RUST_EXECUTABLE, os.X_OK):
                    raise RuntimeError(f"Private vllm-rs is missing: {_RUST_EXECUTABLE}")
                return _RUST_EXECUTABLE
            return original()

        # The envs registry resolves this function at read time. Respect explicit
        # overrides and leave disabled processes and child environments untouched.
        module._resolve_rust_cli_path = resolve


class _NativeFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        binary = _NATIVE.get(fullname)
        if binary:
            return importlib.util.spec_from_file_location(fullname, binary)
        if fullname == "vllm_ascend.utils":
            return importlib.util.spec_from_file_location(fullname, _SHARED_UTILS,
                    loader=_AscendUtils(fullname, _SHARED_UTILS))
        if fullname == "vllm.envs":
            return importlib.util.spec_from_file_location(fullname, _SHARED_VLLM_ENVS,
                    loader=_RustEnv(fullname, _SHARED_VLLM_ENVS))
        return None


def install():
    if _NativeFinder not in sys.meta_path:
        sys.meta_path.insert(0, _NativeFinder)
