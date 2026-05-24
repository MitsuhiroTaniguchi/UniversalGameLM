# UniversalGameLM

UniversalGameLM tokenizes high-level game records into a shared move-token language.

The repository has two modes:

- `demo.py`: small seed-corpus smoke test and gap report.
- `build_production.py`: streaming production shard builder for 3B-token/game runs.

## Production Target

The default target is `3_000_000_000` move tokens per game. A game is not marked ready until its token count reaches that target.

Primary source plan lives in `source_catalog.json`.

Install parser/runtime dependencies before production runs:

```bash
python3 -m pip install -r requirements.txt
```

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
- chess PGN parser errors and illegal move streams are rejected
- shogi requires valid USI moves and explicit terminal reason tokens
- Go requires `SZ:*`, in-range coordinates, and rejects SGF variations
- Othello move streams are replayed against an 8x8 board for legality
- poker private hole-card deal tokens are rejected, including canonical PHH `d dh`
- shards are written to temporary files, atomically renamed, and reported with checksums before upload/delete

Supported production input forms include:

- chess: `.pgn`, `.pgn.gz`, `.pgn.zst`, or directories containing those files
- shogi: `.csa`, `.csa.xz`, `.7z` with `py7zr`, or directories containing `.csa`/`.csa.xz`
- Go: `.sgf`, `.sgf.gz`, or directories containing those files
- Othello: PGN/WTHOR text, local JSONL rows with `moves`/`games`/`seqs`, or `hf://dataset_id[:split]`
- poker: `.phh`, `.phhs`, or directories containing those files

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

## Poker Views

Poker PHH ingestion emits one dataset row per view, matching the MahjongLM
pattern where each row is one tokenized view of one game:

- `view_complete`: public/recorded hand history only. Private `d dh` hole-card deals are excluded.
- `view_imperfect_pN`: player `pN` perspective. It adds exactly that player's private cards to the complete public stream.
- `view_omniscient`: complete stream plus all player private cards and a deterministic sampled full-deck completion.

`view_omniscient` is a consistent completion of the observed hand history. Known
cards from PHH are fixed; unknown private cards and the remaining deck are sampled
without replacement using `uniform_unknown_cards_v1`, with `completion_seed`
stored in metadata for reproducibility.
