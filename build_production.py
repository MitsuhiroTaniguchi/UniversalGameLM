import argparse
import json
import os

from src.production_pipeline import (
    DEFAULT_TARGET_TOKENS,
    build_game_shards,
    maybe_hf_uploader,
)


def main():
    parser = argparse.ArgumentParser(description="Build production token shards for UniversalGameLM.")
    parser.add_argument("--game", required=True, choices=["chess", "shogi", "go", "othello", "poker"])
    parser.add_argument("--input", action="append", required=True, help="Input file or directory. Repeat for multiple sources.")
    parser.add_argument("--output-dir", default="production_shards")
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--max-tokens-per-shard", type=int, default=5_000_000)
    parser.add_argument("--max-records", type=int, default=None, help="Testing cap. Omit for production.")
    parser.add_argument("--hf-repo-id", default=os.environ.get("HF_REPO_ID"))
    parser.add_argument("--hf-repo-prefix", default="")
    parser.add_argument("--delete-after-upload", action="store_true")
    args = parser.parse_args()

    uploader = maybe_hf_uploader(args.hf_repo_id)
    result = build_game_shards(
        args.game,
        args.input,
        args.output_dir,
        target_tokens=args.target_tokens,
        max_tokens_per_shard=args.max_tokens_per_shard,
        max_records=args.max_records,
        uploader=uploader,
        delete_after_upload=args.delete_after_upload,
        repo_prefix=args.hf_repo_prefix or args.game,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
