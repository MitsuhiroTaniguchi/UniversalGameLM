import gzip
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from src.chess_parser import parse_chess_inputs
from src.shogi_parser import parse_shogi_directory
from src.go_parser import parse_go_directory
from src.othello_parser import (
    parse_othello_hf_dataset,
    parse_othello_jsonl_to_tokens,
    parse_othello_pgn_to_tokens,
    validate_othello_moves,
)
from src.poker_parser import parse_phh_to_tokens
from src.hf_uploader import HuggingFaceShardUploader
from src.mahjonglm_compat import entry_to_mahjonglm_row, entry_to_mahjonglm_row_tokens
from src.stats import DatasetStatsAccumulator
from src.tokenizer import UniversalGameTokenizer


GAME_ORDER = ("chess", "shogi", "go", "othello", "poker")
DEFAULT_TARGET_TOKENS = 3_000_000_000
PRIVATE_POKER_TOKEN_PATTERNS = (
    re.compile(r"^h:", re.IGNORECASE),
    re.compile(r"^hole", re.IGNORECASE),
    re.compile(r"^deal[_: -]?hole", re.IGNORECASE),
    re.compile(r"^d[_: -]?dh(?:[_: -]|$)", re.IGNORECASE),
    re.compile(r"^dh(?:[_: -]|$)", re.IGNORECASE),
    re.compile(r"^show[_: -]?or[_: -]?muck[_: -]?hole", re.IGNORECASE),
)


class ProductionDatasetError(RuntimeError):
    pass


@lru_cache(maxsize=4096)
def source_id_for_path(path):
    path_obj = Path(path)
    if str(path).startswith("hf://"):
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        return f"{path}:{digest}"
    resolved = path_obj.resolve()
    hasher = hashlib.sha256()
    if resolved.is_file():
        with open(resolved, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()[:16]
        return f"local:{resolved.name}:sha256:{digest}"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    return f"local-dir:{resolved.name}:{digest}"


def iter_game_entries(game, input_paths, max_records=None):
    emitted = 0
    for input_path in input_paths:
        path = Path(input_path)
        if game == "chess":
            iterator = parse_chess_inputs(str(path), max_games=max_records)
        elif game == "shogi":
            iterator = parse_shogi_directory(str(path), max_games=max_records)
        elif game == "go":
            iterator = parse_go_directory(str(path), max_games=max_records)
        elif game == "othello":
            input_text = str(input_path)
            if input_text.startswith("hf://"):
                dataset_spec = input_text.removeprefix("hf://")
                dataset_id, _, split = dataset_spec.partition(":")
                iterator = parse_othello_hf_dataset(dataset_id, split=split or "train", max_games=max_records)
            elif path.suffix.lower() == ".jsonl":
                iterator = parse_othello_jsonl_to_tokens(str(path), max_games=max_records)
            else:
                iterator = parse_othello_pgn_to_tokens(str(path), max_games=max_records)
        elif game == "poker":
            iterator = parse_phh_to_tokens(str(path), max_hands=max_records)
        else:
            raise ValueError(f"Unsupported game: {game}")

        for tokens, metadata in iterator:
            actual_source = metadata.get("source_path") or str(input_path)
            emitted += 1
            yield {
                "game": game,
                "tokens": tokens,
                "metadata": {
                    **metadata,
                    "source_id": source_id_for_path(actual_source),
                    "source_name": Path(actual_source).name if not str(actual_source).startswith("hf://") else actual_source,
                    "ingestion_version": 2,
                },
            }
            if max_records and emitted >= max_records and game != "poker":
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
    if game == "chess":
        for token in tokens[2:-1]:
            if token.startswith(("FEN:", "VARIANT:")):
                continue
            if not re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", token):
                raise ProductionDatasetError(f"Invalid chess token: {token}")
    if game == "shogi":
        for token in tokens[2:-1]:
            if token.startswith(("SETUP:", "TURN:", "END:")):
                continue
            if token == "None" or not re.fullmatch(r"(?:[1-9][a-i][1-9][a-i]\+?|[PLNSGBR]\*[1-9][a-i])", token):
                raise ProductionDatasetError(f"Invalid shogi USI token: {token}")
        if not any(token.startswith("END:") for token in tokens):
            raise ProductionDatasetError("Shogi entry is missing terminal token")
    if game == "go":
        if not tokens[2].startswith("SZ:"):
            raise ProductionDatasetError("Go entry is missing board size token")
        board_size = int(tokens[2].split(":", 1)[1])
        for token in tokens[3:-1]:
            if token.endswith(":pass"):
                continue
            if token.startswith(("AB:", "AW:", "AE:", "b:", "w:")):
                point = token.split(":", 1)[1]
                if not re.fullmatch(r"[a-z]{2}", point):
                    raise ProductionDatasetError(f"Invalid Go point token: {token}")
                col = ord(point[0]) - ord("a")
                row = ord(point[1]) - ord("a")
                if not (0 <= col < board_size and 0 <= row < board_size):
                    raise ProductionDatasetError(f"Go point out of range: {token}")
                continue
            raise ProductionDatasetError(f"Invalid Go token: {token}")
    if game == "othello":
        try:
            validate_othello_moves(tokens[2:-1])
        except ValueError as exc:
            raise ProductionDatasetError(f"Invalid Othello sequence: {exc}") from exc
    if game == "poker":
        if not tokens[2].startswith("view_"):
            raise ProductionDatasetError("Poker entry is missing a view token")
        if any(pattern.search(token) for token in tokens for pattern in PRIVATE_POKER_TOKEN_PATTERNS):
            raise ProductionDatasetError("Poker entry leaks raw private hole-card tokens")
        view_type = (entry.get("metadata") or {}).get("view_type")
        if tokens[2] == "view_complete":
            if view_type not in (None, "complete"):
                raise ProductionDatasetError("Poker complete view metadata mismatch")
            if any(token.startswith(("private_cards:", "deck:")) for token in tokens):
                raise ProductionDatasetError("Poker complete view contains hidden-card tokens")
        elif tokens[2].startswith("view_imperfect_p"):
            viewer = tokens[2].removeprefix("view_imperfect_")
            private_tokens = [token for token in tokens if token.startswith("private_cards:")]
            if len(private_tokens) != 1 or not private_tokens[0].startswith(f"private_cards:{viewer}:"):
                raise ProductionDatasetError("Poker imperfect view must contain exactly the viewer's private cards")
            if any(token.startswith("deck:") for token in tokens):
                raise ProductionDatasetError("Poker imperfect view cannot contain deck tokens")
        elif tokens[2] == "view_omniscient":
            if not any(token.startswith("deck:") for token in tokens):
                raise ProductionDatasetError("Poker omniscient view is missing sampled deck token")
        else:
            raise ProductionDatasetError(f"Unknown poker view token: {tokens[2]}")


def row_token_count(row):
    if "tokens" in row:
        return len(row["tokens"])
    if "input_ids" in row:
        return len(row["input_ids"])
    return int(row.get("length") or 0)


class JsonlShardWriter:
    def __init__(self, output_dir, game, max_tokens_per_shard=5_000_000, compress=True, row_transform=None):
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
        self.current_temp_path = None
        self.current_hash = None
        self.completed = []
        self.row_transform = row_transform

    def _open_next(self):
        suffix = ".jsonl.gz" if self.compress else ".jsonl"
        self.current_path = self.output_dir / f"{self.game}-{self.shard_index:06d}{suffix}"
        self.current_temp_path = self.output_dir / f".{self.game}-{self.shard_index:06d}{suffix}.tmp"
        if self.current_path.exists():
            raise ProductionDatasetError(
                f"Refusing to overwrite existing shard: {self.current_path}. "
                "Use a new output directory or implement an explicit resume manifest."
            )
        if self.current_temp_path.exists():
            raise ProductionDatasetError(f"Refusing to overwrite existing temp shard: {self.current_temp_path}")
        self.current_raw_file = open(self.current_temp_path, "wb")
        self.current_file = gzip.GzipFile(fileobj=self.current_raw_file, mode="wb") if self.compress else self.current_raw_file
        self.current_tokens = 0
        self.current_rows = 0
        self.current_hash = hashlib.sha256()
        self.shard_index += 1

    def _close_current(self):
        if self.current_file is None:
            return None
        self.current_file.close()
        if self.compress and self.current_raw_file is not None:
            self.current_raw_file.close()
        os.replace(self.current_temp_path, self.current_path)
        info = {
            "path": str(self.current_path),
            "tokens": self.current_tokens,
            "rows": self.current_rows,
            "sha256_uncompressed_jsonl": self.current_hash.hexdigest(),
        }
        self.completed.append(info)
        self.current_file = None
        self.current_raw_file = None
        self.current_path = None
        self.current_temp_path = None
        self.current_hash = None
        return info

    def write(self, entry):
        validate_entry(entry)
        row = self.row_transform(entry) if self.row_transform else entry
        if self.current_file is None:
            self._open_next()

        token_count = row_token_count(row)
        if self.current_rows > 0 and self.current_tokens + token_count > self.max_tokens_per_shard:
            completed = self._close_current()
            self._open_next()
        else:
            completed = None

        payload = json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
        self.current_hash.update(payload)
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
    output_format="universal_jsonl",
    tokenizer=None,
):
    if output_format not in {"universal_jsonl", "mahjonglm_jsonl"}:
        raise ValueError(f"Unsupported output_format: {output_format}")
    if output_format == "mahjonglm_jsonl" and tokenizer is None:
        raise ValueError("mahjonglm_jsonl output requires a tokenizer")

    row_transform = None
    if output_format == "mahjonglm_jsonl":
        row_transform = lambda entry: entry_to_mahjonglm_row(entry, tokenizer)

    writer = JsonlShardWriter(
        output_dir,
        game,
        max_tokens_per_shard=max_tokens_per_shard,
        row_transform=row_transform,
    )
    stats = DatasetStatsAccumulator()
    total_tokens = 0
    total_rows = 0
    uploaded = []

    try:
        for entry in iter_game_entries(game, input_paths, max_records=max_records):
            metadata = entry.get("metadata") or {}
            starts_new_view_group = game != "poker" or metadata.get("view_type") == "complete"
            if total_tokens >= target_tokens and starts_new_view_group:
                break
            stats_entry = entry
            if output_format == "mahjonglm_jsonl":
                stats_entry = {**entry, "tokens": entry_to_mahjonglm_row_tokens(entry)}
            completed = writer.write(entry)
            stats.update(stats_entry)
            total_tokens += row_token_count(row_transform(entry) if row_transform else entry)
            total_rows += 1

            if completed and uploader:
                repo_path = str(Path(repo_prefix) / Path(completed["path"]).name)
                uploader.upload_file(completed["path"], repo_path, delete_local=delete_after_upload)
                uploaded.append(repo_path)

        final = writer.close()
    except Exception:
        if writer.current_file is not None:
            try:
                writer.current_file.close()
            finally:
                if writer.compress and writer.current_raw_file is not None:
                    writer.current_raw_file.close()
                if writer.current_temp_path and Path(writer.current_temp_path).exists():
                    Path(writer.current_temp_path).unlink()
        raise
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


def build_mahjonglm_tokenizer(base_tokenizer_dir, input_specs, output_dir):
    """
    Extends a MahjongLM tokenizer with tokens needed by UniversalGameLM entries.

    input_specs: iterable of (game, [input_paths], max_records)
    """
    tokenizer = UniversalGameTokenizer.from_mahjonglm_assets(base_tokenizer_dir)
    for game, input_paths, max_records in input_specs:
        for entry in iter_game_entries(game, input_paths, max_records=max_records):
            tokenizer.add_tokens(entry_to_mahjonglm_row_tokens(entry))
    tokenizer.save_mahjonglm_assets(output_dir)
    return tokenizer
