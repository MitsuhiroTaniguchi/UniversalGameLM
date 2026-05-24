# UniversalGameLM

UniversalGameLM tokenizes high-level game records into a shared move-token language.

The repository has two modes:

- `demo.py`: small seed-corpus smoke test and gap report.
- `build_production.py`: streaming production shard builder for 3B-token/game runs.

## Production Target

The default target is `3_000_000_000` move tokens per game. A game is not marked ready until its token count reaches that target.

Primary source plan lives in `source_catalog.json`.

## Build Shards

Example:

```bash
python3 build_production.py \
  --game shogi \
  --input /path/to/aobazero_csa_dir \
  --output-dir production_shards/shogi \
  --target-tokens 3000000000
```

To upload completed shards to Hugging Face and remove local copies after successful upload:

```bash
export HF_TOKEN='...'
export HF_REPO_ID='your-name/universal-game-lm-dataset'
python3 build_production.py \
  --game shogi \
  --input /path/to/aobazero_csa_dir \
  --output-dir production_shards/shogi \
  --hf-repo-id "$HF_REPO_ID" \
  --delete-after-upload
```

## Safety Checks

The production path validates every serialized row:

- sequence starts with `<bos>` and ends with `<eos>`
- game marker matches the selected game
- empty/non-string tokens are rejected
- poker private hole-card tokens are rejected
- shard files are closed before upload/delete

Run tests before any production job:

```bash
python3 -m unittest tests/test_parsers.py
python3 -m py_compile demo.py build_production.py src/*.py
```

## Source Classes

Each dataset source should be labeled in downstream manifests:

- `human_top`
- `human_public`
- `engine_top`
- `engine_public`
- `engine_self_play`
- `generated_or_curated`

Do not merge source classes silently; training can weight them differently.
