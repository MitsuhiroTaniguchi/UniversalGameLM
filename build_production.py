import argparse
import contextlib
import json
import os
import sys

from src.production_pipeline import (
    DEFAULT_TARGET_TOKENS,
    build_game_shards,
    maybe_hf_uploader,
)
from src.tokenizer import UniversalGameTokenizer


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
    parser.add_argument("--output-format", choices=["universal_jsonl", "mahjonglm_jsonl"], default="universal_jsonl")
    parser.add_argument("--mahjonglm-tokenizer-dir", default=os.environ.get("MAHJONGLM_TOKENIZER_DIR"))
    parser.add_argument("--tokenizer-output-dir", default=os.environ.get("TOKENIZER_OUTPUT_DIR"))
    args = parser.parse_args()

    tokenizer = None
    if args.output_format == "mahjonglm_jsonl":
        if not args.mahjonglm_tokenizer_dir:
            raise RuntimeError("--mahjonglm-tokenizer-dir is required for mahjonglm_jsonl output")
        tokenizer = UniversalGameTokenizer.from_mahjonglm_assets(args.mahjonglm_tokenizer_dir)
        from src.mahjonglm_compat import collect_tokens_for_mahjonglm
        from src.production_pipeline import iter_game_entries

        with contextlib.redirect_stdout(sys.stderr):
            for entry in iter_game_entries(args.game, args.input, max_records=args.max_records):
                tokenizer.add_tokens(collect_tokens_for_mahjonglm([entry]))
        if args.tokenizer_output_dir:
            tokenizer.save_mahjonglm_assets(args.tokenizer_output_dir)

    uploader = maybe_hf_uploader(args.hf_repo_id)
    with contextlib.redirect_stdout(sys.stderr):
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
            output_format=args.output_format,
            tokenizer=tokenizer,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
