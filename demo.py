import os
import json
from datetime import datetime, timezone

from src.download import download_chess_pgn, download_shogi_daily, download_go_sgf, download_othello_pgn
from src.chess_parser import parse_pgn_to_tokens
from src.shogi_parser import parse_shogi_directory
from src.go_parser import parse_go_directory
from src.othello_parser import parse_othello_pgn_to_tokens
from src.poker_parser import generate_poker_dataset
from src.tokenizer import UniversalGameTokenizer
from src.stats import DatasetStatsAccumulator
from src.hf_uploader import HuggingFaceShardUploader

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZED_DIR = os.path.join(BASE_DIR, "tokenized_data")
os.makedirs(TOKENIZED_DIR, exist_ok=True)

DATASET_PATH = os.path.join(TOKENIZED_DIR, "dataset.jsonl")
VOCAB_PATH = os.path.join(TOKENIZED_DIR, "vocab.json")
MANIFEST_PATH = os.path.join(TOKENIZED_DIR, "manifest.json")

DEFAULT_TARGET_TOKENS_PER_GAME = 3_000_000_000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<chess>", "<shogi>", "<go>", "<othello>", "<poker>", "<bridge>"]

def env_int(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value.replace("_", ""))

def env_optional_int(name, default=None):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    parsed = int(value.replace("_", ""))
    return None if parsed <= 0 else parsed

def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}

def download_sources():
    """Fetches the small built-in seed corpora used by this repository."""
    shogi_download_max = env_int("SHOGI_DOWNLOAD_MAX_GAMES", 100)
    go_download_max = env_int("GO_DOWNLOAD_MAX_GAMES", 100)
    othello_download_years = env_int("OTHELLO_DOWNLOAD_YEARS", 1)
    shogi_date = os.environ.get("SHOGI_DOWNLOAD_DATE", "2026/05/23")

    print("\n--- Step 1: Downloading seed game records ---")
    download_chess_pgn(os.environ.get("CHESS_PLAYER", "Carlsen"))
    download_shogi_daily(shogi_date, max_games=shogi_download_max)
    download_go_sgf(max_games=go_download_max)
    download_othello_pgn(max_years=othello_download_years)

def iter_real_game_entries():
    """Streams parsed real-game entries from local source files."""
    chess_player = os.environ.get("CHESS_PLAYER", "Carlsen")
    chess_pgn_path = os.path.join(BASE_DIR, "data", "chess", f"{chess_player}.pgn")
    chess_max = env_optional_int("CHESS_PARSE_MAX_GAMES", 100)
    if os.path.exists(chess_pgn_path):
        for tokens, meta in parse_pgn_to_tokens(chess_pgn_path, max_games=chess_max):
            yield {"game": "chess", "tokens": tokens, "metadata": meta}

    shogi_dir = os.path.join(BASE_DIR, "data", "shogi")
    shogi_max = env_optional_int("SHOGI_PARSE_MAX_GAMES", None)
    if os.path.exists(shogi_dir):
        for tokens, meta in parse_shogi_directory(shogi_dir, max_games=shogi_max):
            yield {"game": "shogi", "tokens": tokens, "metadata": meta}

    go_dir = os.path.join(BASE_DIR, "data", "go")
    go_max = env_optional_int("GO_PARSE_MAX_GAMES", None)
    if os.path.exists(go_dir):
        for tokens, meta in parse_go_directory(go_dir, max_games=go_max):
            yield {"game": "go", "tokens": tokens, "metadata": meta}

    othello_pgn_path = os.path.join(BASE_DIR, "data", "othello", "WTH_2024.pgn")
    othello_max = env_optional_int("OTHELLO_PARSE_MAX_GAMES", 100)
    if os.path.exists(othello_pgn_path):
        for tokens, meta in parse_othello_pgn_to_tokens(othello_pgn_path, max_games=othello_max):
            yield {"game": "othello", "tokens": tokens, "metadata": meta}

def iter_synthetic_entries():
    """Streams optional simulator-only entries that must not be mixed into default real-data runs."""
    if not env_bool("INCLUDE_SYNTHETIC_POKER"):
        return

    n_hands = env_int("POKER_SYNTHETIC_HANDS", 100)
    for tokens, meta in generate_poker_dataset(n_hands=n_hands):
        yield {"game": "poker", "tokens": tokens, "metadata": meta}

def iter_dataset_entries():
    yield from iter_real_game_entries()
    yield from iter_synthetic_entries()

def build_manifest(stats_summary, target_tokens_per_game):
    games = {}
    for game, item in stats_summary.items():
        coverage = item["coverage"] or 0
        games[game] = {
            "games": item["games"],
            "tokens": item["tokens"],
            "target_tokens": item["target_tokens"],
            "token_deficit": item["token_deficit"],
            "coverage": coverage,
            "status": "ready" if item["token_deficit"] == 0 else "insufficient",
        }

    missing_games = sorted(set(["chess", "shogi", "go", "othello", "poker", "bridge"]) - set(games))
    for game in missing_games:
        games[game] = {
            "games": 0,
            "tokens": 0,
            "target_tokens": target_tokens_per_game,
            "token_deficit": target_tokens_per_game,
            "coverage": 0,
            "status": "missing",
        }

    complete = all(item["status"] == "ready" for item in games.values())
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": DATASET_PATH,
        "vocab_path": VOCAB_PATH,
        "target_tokens_per_game": target_tokens_per_game,
        "status": "ready" if complete else "sample_insufficient_for_training",
        "games": games,
        "notes": [
            "Default output is a seed corpus and quality check artifact, not a 3B-token-per-game training corpus.",
            "Set *_PARSE_MAX_GAMES=0 to remove parser caps for locally available files.",
            "Set STRICT_TOKEN_TARGETS=1 to fail the run when any game is below target.",
        ],
    }

def main():
    print("="*70)
    print("      UniversalGameLM Dataset Orchestration Pipeline")
    print("="*70)

    target_tokens_per_game = env_int("TARGET_TOKENS_PER_GAME", DEFAULT_TARGET_TOKENS_PER_GAME)
    print(f"Target per game: {target_tokens_per_game:,} tokens")

    download_sources()

    print("\n--- Step 2: Collecting Parsed Entries ---")
    entries = list(iter_dataset_entries())
    if not entries:
        raise RuntimeError("No games parsed; dataset was not created.")
    print(f"Collected {len(entries)} tokenized games for deterministic vocab/serialization.")

    print("\n--- Step 3: Building Shared Universal Vocabulary ---")
    tokenizer = UniversalGameTokenizer(special_tokens=SPECIAL_TOKENS)
    tokenizer.build_vocab(entry["tokens"] for entry in entries)
    tokenizer.save_vocab(VOCAB_PATH)

    print("\n--- Step 4: Serializing Dataset ---")
    stats = DatasetStatsAccumulator()
    row_count = 0
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for entry in entries:
            encoded_ids = tokenizer.encode(entry["tokens"])
            serialized = {
                "game": entry["game"],
                "tokens": entry["tokens"],
                "ids": encoded_ids,
                "metadata": entry["metadata"],
            }
            stats.update(serialized)
            row_count += 1
            f.write(json.dumps(serialized, ensure_ascii=False) + "\n")

    print(f"Serialized {row_count} tokenized games to {DATASET_PATH}")

    print("\n--- Step 5: Compiling Dataset Statistics and 3B-Token Gap Report ---")
    stats.print_report(target_tokens_per_game=target_tokens_per_game)
    manifest = build_manifest(stats.summary(target_tokens_per_game), target_tokens_per_game)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved dataset manifest to {MANIFEST_PATH}")

    if env_bool("HF_UPLOAD"):
        repo_id = os.environ.get("HF_REPO_ID")
        if not repo_id:
            raise RuntimeError("Set HF_REPO_ID when HF_UPLOAD=1.")

        delete_after_upload = env_bool("HF_DELETE_LOCAL_AFTER_UPLOAD")
        allow_incomplete_upload = env_bool("HF_ALLOW_INCOMPLETE_UPLOAD")
        if manifest["status"] != "ready" and not allow_incomplete_upload:
            raise RuntimeError(
                "Refusing to upload an incomplete dataset. "
                "Set HF_ALLOW_INCOMPLETE_UPLOAD=1 only for explicit seed-corpus uploads."
            )

        private_repo = env_bool("HF_PRIVATE", default=True)
        print(f"\n--- Step 6: Uploading artifacts to Hugging Face dataset repo {repo_id} ---")
        uploader = HuggingFaceShardUploader(repo_id)
        uploaded = uploader.upload_directory_files(
            TOKENIZED_DIR,
            repo_prefix=os.environ.get("HF_REPO_PREFIX", ""),
            delete_local=delete_after_upload,
            private=private_repo,
        )
        print(f"Uploaded {len(uploaded)} artifacts.")
        if delete_after_upload:
            print("Deleted uploaded local artifacts after successful uploads.")

    if manifest["status"] != "ready":
        print("\n[Warning] Dataset is below the requested per-game token target.")
        for game, item in sorted(manifest["games"].items()):
            if item["status"] != "ready":
                print(f"  {game}: {item['tokens']:,}/{item['target_tokens']:,} tokens "
                      f"({item['coverage']:.8%}); deficit {item['token_deficit']:,}")
        if env_bool("STRICT_TOKEN_TARGETS"):
            raise RuntimeError("STRICT_TOKEN_TARGETS=1 and at least one game is below target.")

    print("\n" + "="*70)
    print("      UniversalGameLM Dataset Orchestration Complete")
    print("="*70)

if __name__ == "__main__":
    main()
