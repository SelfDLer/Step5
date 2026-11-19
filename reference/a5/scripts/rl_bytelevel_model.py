"""Select a Step5 model directory whose HF loader uses its released ByteLevel backend."""
import argparse
import errno
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def rl_model_path(model, run_dir):
    source = Path(model).resolve()
    config = source / "tokenizer_config.json"
    if not config.is_file():
        return model
    settings = json.loads(config.read_text())
    if settings.get("tokenizer_class") not in ("LlamaTokenizer", "LlamaTokenizerFast"):
        return model
    if json.loads((source / "config.json").read_text()).get("model_type") != "step5_vl":
        return model
    if json.loads((source / "tokenizer.json").read_text())["decoder"]["type"] != "ByteLevel":
        return model

    settings["tokenizer_class"] = "TokenizersBackend"
    metadata = json.dumps(settings, ensure_ascii=False, indent=2) + "\n"
    parent = Path(run_dir).resolve()
    if parent.is_relative_to(source):
        raise ValueError("The RL output directory must be outside its source checkpoint")
    parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(source.iterdir())
    identity = json.dumps([str(source), metadata, [entry.name for entry in entries]])
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    target = parent / f"step5-bytelevel-{suffix}"
    if not target.exists():
        with TemporaryDirectory(dir=parent) as temporary:
            stage = Path(temporary) / "model"
            stage.mkdir()
            for entry in entries:
                if entry.name == config.name:
                    (stage / entry.name).write_text(metadata, encoding="utf-8")
                else:
                    (stage / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
            try:
                stage.rename(target)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
    if (target / config.name).read_text(encoding="utf-8") != metadata or any(
        not (target / entry.name).is_symlink() or (target / entry.name).readlink() != entry
        for entry in entries if entry != config
    ):
        raise RuntimeError(f"Inconsistent Step5 RL model directory: {target}")
    return str(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    print(rl_model_path(args.model, args.run_dir))
