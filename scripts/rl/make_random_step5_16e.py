"""Create a seeded FP32 Hugging Face checkpoint for the fixed Step5 16-expert layout."""
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
        end = offset + prod(shape) * 4
        header[name] = dict(dtype="F32", shape=shape, data_offsets=[offset, end])
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
                out.write(memoryview(data.astype("<f4", copy=False)))
                remaining -= n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path, help="matching config and tokenizer directory")
    parser.add_argument("--output", required=True, type=Path, help="new checkpoint directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plan", action="store_true", help="validate layout and print size only")
    args = parser.parse_args()
    assets = args.assets.resolve()
    dest = args.output.resolve()
    layout = json.loads(Path(__file__).with_name("step5_16e_layout.json").read_text())
    config = json.loads((assets / "config.json").read_text())
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != layout["config_sha256"]:
        parser.error("config.json differs from the sft100 16-expert baseline")
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    for file in required:
        if not (assets / file).is_file():
            parser.error(f"missing {assets / file}")
    shards = defaultdict(list)
    weight_map = {}
    total = 0
    for name, (shard, dtype, shape) in layout["tensors"].items():
        if dtype != "F32" or Path(shard).name != shard:
            raise ValueError(f"invalid layout entry for {name}")
        shards[shard].append((name, shape))
        weight_map[name] = shard
        total += 4 * prod(shape)
    print(f"16 experts, top-8, {len(weight_map)} tensors, {len(shards)} shards, {total / 2**30:.1f} GiB, seed={args.seed}", flush=True)
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
        for file in (*required, "chat_template.jinja", "generation_config.json", "special_tokens_map.json",
                     "added_tokens.json", "preprocessor_config.json", "processor_config.json"):
            if (assets / file).is_file():
                shutil.copyfile(assets / file, stage / file)
        for i, (shard, tensors) in enumerate(sorted(shards.items()), 1):
            write_shard(stage / shard, tensors, args.seed)
            print(f"[{i}/{len(shards)}] {shard}", flush=True)
        index = {"metadata": {"total_size": total}, "weight_map": weight_map}
        (stage / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        (stage / "random-init.json").write_text(json.dumps({"seed": args.seed, "numpy": np.__version__, "config_sha256": digest}))
        stage.rename(dest)
    print(f"MODEL={dest}")


if __name__ == "__main__":
    main()
