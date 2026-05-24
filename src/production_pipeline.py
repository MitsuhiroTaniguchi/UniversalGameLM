import gzip
import json
import os
from pathlib import Path

from src.chess_parser import parse_pgn_to_tokens
from src.shogi_parser import parse_shogi_directory
from src.go_parser import parse_go_directory
from src.othello_parser import parse_othello_pgn_to_tokens
from src.poker_parser import parse_phh_to_tokens
from src.hf_uploader import HuggingFaceShardUploader
from src.stats import DatasetStatsAccumulator


GAME_ORDER = ("chess", "shogi", "go", "othello", "poker")
DEFAULT_TARGET_TOKENS = 3_000_000_000


class ProductionDatasetError(RuntimeError):
    pass


def iter_game_entries(game, input_paths, max_records=None):
    emitted = 0
    for input_path in input_paths:
        path = Path(input_path)
        if game == "chess":
            iterator = parse_pgn_to_tokens(str(path), max_games=max_records)
        elif game == "shogi":
            iterator = parse_shogi_directory(str(path), max_games=max_records)
        elif game == "go":
            iterator = parse_go_directory(str(path), max_games=max_records)
        elif game == "othello":
            iterator = parse_othello_pgn_to_tokens(str(path), max_games=max_records)
        elif game == "poker":
            iterator = parse_phh_to_tokens(str(path), max_hands=max_records)
        else:
            raise ValueError(f"Unsupported game: {game}")

        for tokens, metadata in iterator:
            emitted += 1
            yield {
                "game": game,
                "tokens": tokens,
                "metadata": {
                    **metadata,
                    "source_path": str(path),
                },
            }
            if max_records and emitted >= max_records:
                return


def validate_entry(entry):
    tokens = entry.get("tokens") or []
    game = entry.get("game")
    if game not in GAME_ORDER:
        raise ProductionDatasetError(f"Unknown game in entry: {game}")
    if len(tokens) < 4:
        raise ProductionDatasetError(f"{game} sequence is too short: {tokens}")
    if tokens[0] != "<bos>" or tokens[-1] != "<eos>":
        raise ProductionDatasetError(f"{game} sequence must start/end with BOS/EOS")
    if tokens[1] != f"<{game}>":
        raise ProductionDatasetError(f"{game} sequence has wrong game marker: {tokens[1]}")
    if any(not isinstance(token, str) or not token for token in tokens):
        raise ProductionDatasetError(f"{game} sequence contains an invalid token")
    if game == "poker" and any(token.startswith("h:") or token.startswith("hole") for token in tokens):
        raise ProductionDatasetError("Poker entry leaks private hole-card tokens")


class JsonlShardWriter:
    def __init__(self, output_dir, game, max_tokens_per_shard=5_000_000, compress=True):
        self.output_dir = Path(output_dir)
        self.game = game
        self.max_tokens_per_shard = max_tokens_per_shard
        self.compress = compress
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_index = 0
        self.current_tokens = 0
        self.current_rows = 0
        self.current_file = None
        self.current_raw_file = None
        self.current_path = None
        self.completed = []

    def _open_next(self):
        suffix = ".jsonl.gz" if self.compress else ".jsonl"
        self.current_path = self.output_dir / f"{self.game}-{self.shard_index:06d}{suffix}"
        self.current_raw_file = open(self.current_path, "wb")
        self.current_file = gzip.GzipFile(fileobj=self.current_raw_file, mode="wb") if self.compress else self.current_raw_file
        self.current_tokens = 0
        self.current_rows = 0
        self.shard_index += 1

    def _close_current(self):
        if self.current_file is None:
            return None
        self.current_file.close()
        if self.compress and self.current_raw_file is not None:
            self.current_raw_file.close()
        info = {
            "path": str(self.current_path),
            "tokens": self.current_tokens,
            "rows": self.current_rows,
        }
        self.completed.append(info)
        self.current_file = None
        self.current_raw_file = None
        self.current_path = None
        return info

    def write(self, entry):
        validate_entry(entry)
        if self.current_file is None:
            self._open_next()

        token_count = len(entry["tokens"])
        if self.current_rows > 0 and self.current_tokens + token_count > self.max_tokens_per_shard:
            completed = self._close_current()
            self._open_next()
        else:
            completed = None

        payload = json.dumps(entry, ensure_ascii=False).encode("utf-8") + b"\n"
        self.current_file.write(payload)
        self.current_tokens += token_count
        self.current_rows += 1
        return completed

    def close(self):
        return self._close_current()


def build_game_shards(
    game,
    input_paths,
    output_dir,
    target_tokens=DEFAULT_TARGET_TOKENS,
    max_tokens_per_shard=5_000_000,
    max_records=None,
    uploader=None,
    delete_after_upload=False,
    repo_prefix="",
):
    writer = JsonlShardWriter(output_dir, game, max_tokens_per_shard=max_tokens_per_shard)
    stats = DatasetStatsAccumulator()
    total_tokens = 0
    total_rows = 0
    uploaded = []

    for entry in iter_game_entries(game, input_paths, max_records=max_records):
        if total_tokens >= target_tokens:
            break
        completed = writer.write(entry)
        stats.update(entry)
        total_tokens += len(entry["tokens"])
        total_rows += 1

        if completed and uploader:
            repo_path = str(Path(repo_prefix) / Path(completed["path"]).name)
            uploader.upload_file(completed["path"], repo_path, delete_local=delete_after_upload)
            uploaded.append(repo_path)

    final = writer.close()
    if final and uploader:
        repo_path = str(Path(repo_prefix) / Path(final["path"]).name)
        uploader.upload_file(final["path"], repo_path, delete_local=delete_after_upload)
        uploaded.append(repo_path)

    status = "ready" if total_tokens >= target_tokens else "insufficient"
    return {
        "game": game,
        "status": status,
        "rows": total_rows,
        "tokens": total_tokens,
        "target_tokens": target_tokens,
        "token_deficit": max(target_tokens - total_tokens, 0),
        "shards": writer.completed,
        "uploaded": uploaded,
        "stats": stats.summary(target_tokens_per_game=target_tokens).get(game, {}),
    }


def load_source_catalog(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_hf_uploader(repo_id):
    if not repo_id:
        return None
    uploader = HuggingFaceShardUploader(repo_id)
    uploader.ensure_repo(private=os.environ.get("HF_PRIVATE", "1") != "0")
    return uploader
