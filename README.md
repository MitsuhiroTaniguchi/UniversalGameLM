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
  --source-name "AobaZero self-play CSA archives" \
  --output-dir production_shards/shogi \
  --target-tokens 3000000000
```

To build shards that can be concatenated with `mitsutani/mahjonglm-dataset`,
load the MahjongLM tokenizer assets and emit MahjongLM-style rows:

```bash
python3 build_production.py \
  --game poker \
  --input /path/to/phh_dir \
  --source-name "PHH full corpus ACPC public subsets" \
  --output-dir production_shards/poker \
  --output-format mahjonglm_jsonl \
  --mahjonglm-tokenizer-dir /path/to/mahjonglm-dataset/tokenizer \
  --tokenizer-output-dir tokenized_data/universal_tokenizer
```

`mahjonglm_jsonl` rows use the MahjongLM-compatible fields
`game_id`, `year`, `seat_count`, `view_type`, `viewer_seat`, `length`,
and `input_ids`. Boundary tokens are not serialized in `input_ids`; `<bos>` and
`<eos>` remain training-time boundary tokens, matching MahjongLM's convention.
The base MahjongLM token ids are preserved exactly, and new game tokens are
appended at the end of the vocabulary.

To upload completed shards to Hugging Face and remove local copies after successful upload:

```bash
export HF_TOKEN='...'
export HF_REPO_ID='your-name/universal-game-lm-dataset'
python3 build_production.py \
  --game shogi \
  --input /path/to/aobazero_csa_dir \
  --source-name "AobaZero self-play CSA archives" \
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
- poker private hole-card deal actions are excluded from complete views; observed
  private cards are only emitted in the corresponding imperfect/omniscient views
- shards are written to temporary files, atomically renamed, and reported with checksums before upload/delete

Supported production input forms include:

- chess: `.pgn`, `.pgn.gz`, `.pgn.zst`, or directories containing those files
- shogi: `.csa`, `.csa.xz`, `.7z` with `py7zr`, or directories containing `.csa`/`.csa.xz`
- Go: `.sgf`, `.sgf.gz`, or directories containing those files
- Othello: PGN/WTHOR text, local JSONL rows with `moves`/`games`/`seqs`, or `hf://dataset_id[:split]`
- poker: `.phh`, `.phhs`, or directories containing those files
- bridge: `.pbn` files or directories containing `.pbn`

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
For primary 3B-token builds, use only `engine_top` or `human_top` sources unless
the manifest documents a strict top-player/top-engine filter. General public
archives such as raw Lichess, OGS, Floodgate, BBO/Vugraph, and generic online
poker hand histories are fallback/evaluation sources, not default training
sources. `build_production.py` enforces this for uncapped 3B-token builds through
`--source-name`; smoke tests may use `--max-records` without a catalog entry.

Current primary volume plan:

- chess: Stockfish fishtest LTC PGNs, mixed with Lc0 self-play for style diversity; Lichess Elite as human-top mix/evaluation
- shogi: AobaZero self-play; dlshogi-style archives only when concrete license and conversion are verified
- Go: KataGo distributed self-play data; ZhiziGo/KataGo mirrors only after conversion validation
- Othello: Egaroucid/Edax engine self-play, with solved-line and WTHOR sets for seed/evaluation
- poker: ACPC engine subsets from the PHH full corpus; Pluribus as a small top-quality seed/evaluation set
- bridge: generated WBridge5/RoboBridge/OpenSpiel/DDS-assisted data for volume, with ComputerBridge/WBF/ACBL finals as human-top seeds

## Poker Views

Poker PHH ingestion emits one dataset row per view, matching the MahjongLM
pattern where each row is one tokenized view of one game:

- `view_complete`: public/recorded hand history only. Private `d dh` hole-card deals are excluded.
- `view_imperfect_pN`: player `pN` perspective. It adds exactly that player's private cards to the complete public stream.
- `view_omniscient`: emitted only when every inferred active player has observed private cards in the PHH record. It contains those true private cards plus deterministic `undealt_card:*` tokens, not a sampled private-hand completion.

Unknown private hands are not sampled. This avoids teaching the model false
hand/action correlations when a folded or unshown hand is absent from the source
record. Poker numeric amounts are split into shared digit tokens (`AMT:*` for
actions and `NUM:*` for state values) to avoid one token per stack or bet size.

Poker rows also carry MahjongLM-style `seat_count` metadata. A hand with `N`
players emits one complete view, one imperfect view per observed private hand,
and an omniscient view only if all active private hands are observed. Production
stats report `by_seat_count` buckets so two-player, six-player, and larger table
corpora can be balanced explicitly.

## Bridge

Contract bridge ingestion supports PBN records with `Deal`, `Auction`, and
optional `Play` sections. Bridge rows emit `view_complete`, four
`view_imperfect_*` rows, and `view_omniscient`; play tokens include the acting
seat (`play:N:As`) and cards use the same rank+suit notation as poker. The source
plan prioritizes generated top-engine bridge for volume plus top-event PBN seeds.
BBO/Vugraph archives should be used only with explicit event/player filters.
