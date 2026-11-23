"""Create a seeded BF16 Hugging Face checkpoint for the fixed Step5 16-expert layout."""
import argparse
from collections import defaultdict
import hashlib
import json
from math import prod
from pathlib import Path
import shutil
import struct
from tempfile import TemporaryDirectory

import numpy as np


def constant(name, shape):
    if name.endswith((".bias", "_bias", ".ssmax_s")):
        return 0.0
    if name.endswith(".gamma"):
        return 0.1
    if len(shape) == 1 and name.endswith(".weight"):
        return float(name.startswith("vision_model.") or ".sparse_indexer_k_norm." in name)
    return None


def write_shard(path, tensors, seed):
    # safetensors stores a padded JSON header and contiguous little-endian data.
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, shape in tensors:
        end = offset + prod(shape) * 2
        header[name] = dict(dtype="BF16", shape=shape, data_offsets=[offset, end])
        offset = end
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("xb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for name, shape in tensors:
            key = hashlib.sha256(f"{seed}:{name}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(key[:16], "little"))
            fill = constant(name, shape)
            remaining = prod(shape)
            while remaining:
                n = min(remaining, 1 << 20)
                if fill is None:
                    data = rng.standard_normal(n, dtype=np.float32)
                    data *= np.float32(0.02)
                else:
                    data = np.full(n, fill, dtype=np.float32)
                # Round finite FP32 values to BF16, nearest with ties to even.
                bits = data.view(np.uint32)
                rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
                out.write(memoryview((rounded >> 16).astype("<u2")))
                remaining -= n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path, help="Step5 HF config/tokenizer directory; weights need not be downloaded")
    parser.add_argument("--output", required=True, type=Path, help="new checkpoint directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plan", action="store_true", help="validate layout and print size only")
    args = parser.parse_args()
    assets = args.assets.resolve()
    dest = args.output.resolve()
    layout = json.loads(Path(__file__).with_name("step5_16e_layout.json").read_text())
    target_config = Path(__file__).with_name("step5_16e_config.json")
    config = json.loads(target_config.read_text())
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != layout["config_sha256"]:
        parser.error("bundled 16-expert config and tensor layout do not match")
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    def read_asset(name):
        try:
            return json.loads((assets / name).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"missing or incomplete {assets / name}: {exc}")

    source = read_asset("config.json")
    tokenizer = read_asset("tokenizer.json")
    settings = read_asset("tokenizer_config.json")
    if source.get("model_type") != config["model_type"]:
        parser.error(f"expected a {config['model_type']} HF config in --assets")
    vocab_size = config["text_config"]["vocab_size"]
    if source.get("text_config", {}).get("vocab_size") != vocab_size:
        parser.error(f"the 16-expert embedding table expects vocab_size={vocab_size}")
    if source.get("image_token_id") != config.get("image_token_id"):
        parser.error("image_token_id differs from the target Step5 config")
    vocab = tokenizer.get("model", {}).get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        parser.error("expected a complete Step5 tokenizer.json with a BPE vocabulary")
    ids = [*vocab.values(), *(t["id"] for t in tokenizer.get("added_tokens", []))]
    if min(ids) < 0 or max(ids) >= vocab_size:
        parser.error("tokenizer contains IDs outside the target embedding table")
    if not settings.get("chat_template") and not (assets / "chat_template.jinja").is_file():
        print("Note: no chat template found; check the tokenizer assets before RL.")
    shards = defaultdict(list)
    weight_map = {}
    total = 0
    for name, (shard, dtype, shape) in layout["tensors"].items():
        if dtype != "F32" or Path(shard).name != shard:
            raise ValueError(f"invalid layout entry for {name}")
        shards[shard].append((name, shape))
        weight_map[name] = shard
        total += 2 * prod(shape)
    print(f"BF16 checkpoint, 16 experts, top-8, {len(weight_map)} tensors, {len(shards)} shards, {total / 2**30:.1f} GiB, seed={args.seed}", flush=True)
    if args.plan:
        return
    if dest.exists() or dest.is_relative_to(assets):
        parser.error("choose a new output directory outside the assets directory")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(dest.parent).free < total + (1 << 30):
        parser.error("insufficient free space: allow checkpoint size plus 1 GiB")
    with TemporaryDirectory(prefix=".step5-random-", dir=dest.parent) as tmp:
        stage = Path(tmp) / "model"
        stage.mkdir()
        shutil.copyfile(target_config, stage / "config.json")
        for file in (*required[1:], "chat_template.jinja", "generation_config.json", "special_tokens_map.json",
                     "added_tokens.json", "preprocessor_config.json", "processor_config.json"):
            if (assets / file).is_file():
                shutil.copyfile(assets / file, stage / file)
        for i, (shard, tensors) in enumerate(sorted(shards.items()), 1):
            write_shard(stage / shard, tensors, args.seed)
            print(f"[{i}/{len(shards)}] {shard}", flush=True)
        index = {"metadata": {"total_size": total}, "weight_map": weight_map}
        (stage / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        (stage / "random-init.json").write_text(json.dumps({"checkpoint_dtype": "BF16", "seed": args.seed, "numpy": np.__version__, "config_sha256": digest}))
        stage.rename(dest)
    print(f"MODEL={dest}")


if __name__ == "__main__":
    main()
