import gzip
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
import chess

from src.chess_parser import parse_chess_inputs
from src.shogi_parser import parse_shogi_directory, validate_shogi_token_sequence
from src.go_parser import parse_go_directory, validate_go_token_sequence
from src.othello_parser import (
    parse_othello_hf_dataset,
    parse_othello_inputs,
    parse_othello_jsonl_to_tokens,
    parse_othello_pgn_to_tokens,
    validate_othello_moves,
)
from src.poker_parser import parse_phh_to_tokens
from src.bridge_parser import parse_bridge_inputs
from src.bridge_parser import _validate_auction as validate_bridge_auction
from src.bridge_parser import expected_play_seats as bridge_expected_play_seats
from src.bridge_parser import _validate_play as validate_bridge_play
from src.hf_uploader import HuggingFaceShardUploader
from src.mahjonglm_compat import entry_to_mahjonglm_row, entry_to_mahjonglm_row_tokens
from src.stats import DatasetStatsAccumulator
from src.tokenizer import UniversalGameTokenizer


GAME_ORDER = ("chess", "shogi", "go", "othello", "poker", "bridge")
DEFAULT_TARGET_TOKENS = 3_000_000_000
PRIVATE_POKER_TOKEN_PATTERNS = (
    re.compile(r"^h:", re.IGNORECASE),
    re.compile(r"^hole", re.IGNORECASE),
    re.compile(r"^deal[_: -]?hole", re.IGNORECASE),
    re.compile(r"^d[_: -]?dh(?:[_: -]|$)", re.IGNORECASE),
    re.compile(r"^dh(?:[_: -]|$)", re.IGNORECASE),
    re.compile(r"^show[_: -]?or[_: -]?muck[_: -]?hole", re.IGNORECASE),
)
POKER_STREET_ORDER = {
    "act:preflop": 0,
    "act:flop": 1,
    "act:turn": 2,
    "act:river": 3,
}
POKER_PLAYER_ACTIONS = {
    "act:post_small_blind",
    "act:post_big_blind",
    "act:post_blind",
    "act:post_ante",
    "act:blind",
    "act:ante",
    "act:bet",
    "act:call",
    "act:check",
    "act:fold",
    "act:raise",
    "act:show",
    "act:muck",
}
POKER_AMOUNT_REQUIRED_ACTIONS = {
    "act:post_small_blind",
    "act:post_big_blind",
    "act:post_blind",
    "act:post_ante",
    "act:blind",
    "act:ante",
    "act:bet",
    "act:raise",
}
POKER_NON_SEAT_ACTIONS = {
    "act:deal_board",
    "act:hidden",
}


class ProductionDatasetError(RuntimeError):
    pass


def _next_token_has_prefix(tokens, index, prefix):
    return index + 1 < len(tokens) and tokens[index + 1].startswith(prefix)


def validate_poker_public_sequence(tokens):
    current_street = -1
    pending_seat = False
    blinds_seen = False
    for index, token in enumerate(tokens[3:-1], start=3):
        if token.startswith((
            "VARIANT:",
            "STARTING_STACKS:",
            "MIN_BET:",
            "ANTE_TRIMMING_STATUS:",
            "BETTING_TYPE:",
            "NUM:",
            "AMT:",
            "private_card:",
            "undealt_card:",
            "card:",
            "showdown:",
            "winner:",
        )):
            continue
        if token.startswith(("BLINDS_OR_STRADDLES:", "ANTES:")):
            blinds_seen = True
            continue
        if token.startswith("seat:"):
            if not re.fullmatch(r"seat:p\d+", token):
                raise ProductionDatasetError(f"Invalid poker seat token: {token}")
            pending_seat = True
            continue
        if token in POKER_STREET_ORDER:
            street_index = POKER_STREET_ORDER[token]
            if street_index <= current_street:
                raise ProductionDatasetError(f"Poker street token is out of order: {token}")
            current_street = street_index
            pending_seat = False
            continue
        if token in POKER_NON_SEAT_ACTIONS:
            pending_seat = False
            continue
        if token.startswith("act:"):
            if token not in POKER_PLAYER_ACTIONS:
                raise ProductionDatasetError(f"Unknown poker action token: {token}")
            if not pending_seat:
                raise ProductionDatasetError(f"Poker player action is missing a preceding seat token: {token}")
            if token in {"act:bet", "act:raise"} and not blinds_seen:
                raise ProductionDatasetError(f"Poker betting action appears before blind/ante posting: {token}")
            if token in {
                "act:post_small_blind",
                "act:post_big_blind",
                "act:post_blind",
                "act:post_ante",
                "act:blind",
                "act:ante",
            }:
                blinds_seen = True
            if token in POKER_AMOUNT_REQUIRED_ACTIONS and not _next_token_has_prefix(tokens, index, "AMT:"):
                raise ProductionDatasetError(f"Poker action is missing amount tokens: {token}")
            pending_seat = False
            continue
        raise ProductionDatasetError(f"Invalid poker token: {token}")


@lru_cache(maxsize=4096)
def source_id_for_path(path):
    if str(path).startswith("hf://"):
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        return f"{path}:{digest}"
    path_obj = Path(path)
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
    emitted_groups = 0
    grouped_views = game in {"poker", "bridge"}
    for input_path in input_paths:
        if max_records is not None and (emitted_groups if grouped_views else emitted) >= max_records:
            return
        remaining = None
        if max_records is not None:
            remaining = max_records - (emitted_groups if grouped_views else emitted)
        path = Path(input_path)
        if game == "chess":
            iterator = parse_chess_inputs(str(path), max_games=remaining)
        elif game == "shogi":
            iterator = parse_shogi_directory(str(path), max_games=remaining)
        elif game == "go":
            iterator = parse_go_directory(str(path), max_games=remaining)
        elif game == "othello":
            input_text = str(input_path)
            if input_text.startswith("hf://"):
                dataset_spec = input_text.removeprefix("hf://")
                dataset_id, _, split = dataset_spec.partition(":")
                iterator = parse_othello_hf_dataset(dataset_id, split=split or "train", max_games=remaining)
            else:
                iterator = parse_othello_inputs(str(path), max_games=remaining)
        elif game == "poker":
            iterator = parse_phh_to_tokens(str(path), max_hands=remaining)
        elif game == "bridge":
            iterator = parse_bridge_inputs(str(path), max_games=remaining)
        else:
            raise ValueError(f"Unsupported game: {game}")

        for tokens, metadata in iterator:
            actual_source = metadata.get("source_path") or str(input_path)
            emitted += 1
            if grouped_views and (metadata.get("view_type") == "complete" or not metadata.get("view_type")):
                emitted_groups += 1
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
            if max_records is not None and not grouped_views and emitted >= max_records:
                return


def iter_cached_entries(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def limit_entries(entries, game, max_records=None):
    if max_records is None:
        yield from entries
        return
    grouped_views = game in {"poker", "bridge"}
    emitted = 0
    emitted_groups = 0
    allow_current_group = False
    for entry in entries:
        metadata = entry.get("metadata") or {}
        starts_group = grouped_views and (metadata.get("view_type") == "complete" or not metadata.get("view_type"))
        if not grouped_views and emitted >= max_records:
            return
        if grouped_views and starts_group:
            if emitted_groups >= max_records:
                return
            emitted_groups += 1
            allow_current_group = True
        elif grouped_views and not allow_current_group:
            continue
        emitted += 1
        yield entry


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
    metadata = entry.get("metadata") or {}
    if metadata.get("seat_count") is None:
        raise ProductionDatasetError(f"{game} entry is missing metadata.seat_count")
    if metadata.get("view_type") is None:
        raise ProductionDatasetError(f"{game} entry is missing metadata.view_type")
    if game == "chess":
        board = chess.Board()
        variant = None
        for token in tokens[2:-1]:
            if token.startswith("VARIANT:"):
                variant = token.split(":", 1)[1]
                if variant in {"chess960", "chess_960", "fischerandom", "fischer_random"}:
                    board = chess.Board(chess960=True)
                elif variant not in {"chess", "standard"}:
                    raise ProductionDatasetError(f"Unsupported chess variant token: {token}")
                continue
            if token.startswith("FEN:"):
                fen = token.split(":", 1)[1].replace("_", " ")
                try:
                    board = chess.Board(fen, chess960=variant in {"chess960", "chess_960", "fischerandom", "fischer_random"})
                except ValueError as exc:
                    raise ProductionDatasetError(f"Invalid chess FEN token: {token}") from exc
                continue
            if not re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", token):
                raise ProductionDatasetError(f"Invalid chess token: {token}")
            move = chess.Move.from_uci(token)
            if move not in board.legal_moves:
                raise ProductionDatasetError(f"Illegal chess move token: {token}")
            board.push(move)
    if game == "shogi":
        try:
            validate_shogi_token_sequence(tokens)
        except Exception as exc:
            raise ProductionDatasetError(f"Invalid Shogi sequence: {exc}") from exc
    if game == "go":
        try:
            validate_go_token_sequence(tokens)
        except Exception as exc:
            raise ProductionDatasetError(f"Invalid Go sequence: {exc}") from exc
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
        view_type = metadata.get("view_type")
        if tokens[2] == "view_complete":
            if view_type not in (None, "complete"):
                raise ProductionDatasetError("Poker complete view metadata mismatch")
            if any(token.startswith(("private_cards:", "private_card:", "deck:")) for token in tokens):
                raise ProductionDatasetError("Poker complete view contains hidden-card tokens")
        elif tokens[2].startswith("view_imperfect_p"):
            viewer = tokens[2].removeprefix("view_imperfect_")
            private_tokens = [token for token in tokens if token.startswith("private_card:")]
            if not (1 <= len(private_tokens) <= 10) or any(not token.startswith(f"private_card:{viewer}:") for token in private_tokens):
                raise ProductionDatasetError("Poker imperfect view must contain only the viewer's private card tokens")
            if any(token.startswith(("deck:", "undealt_cards:", "undealt_card:")) for token in tokens):
                raise ProductionDatasetError("Poker imperfect view cannot contain deck tokens")
        elif tokens[2] == "view_omniscient":
            if not any(token.startswith("private_card:") for token in tokens):
                raise ProductionDatasetError("Poker omniscient view is missing private-card tokens")
            if not any(token.startswith("undealt_card:") for token in tokens):
                raise ProductionDatasetError("Poker omniscient view is missing undealt-card tokens")
        else:
            raise ProductionDatasetError(f"Unknown poker view token: {tokens[2]}")
        validate_poker_public_sequence(tokens)
    if game == "bridge":
        if not tokens[2].startswith("view_"):
            raise ProductionDatasetError("Bridge entry is missing view token")
        hand_tokens = [token for token in tokens if token.startswith("hand:")]
        if tokens[2] == "view_complete" and hand_tokens:
            raise ProductionDatasetError("Bridge complete view must not contain hidden hands")
        if tokens[2].startswith("view_imperfect_") and len(hand_tokens) != 13:
            raise ProductionDatasetError("Bridge imperfect view must contain exactly one 13-card hand")
        if tokens[2] == "view_omniscient" and len(hand_tokens) != 52:
            raise ProductionDatasetError("Bridge omniscient view must contain 52 hand-card tokens")
        if tokens[2] not in {"view_complete", "view_omniscient"} and not tokens[2].startswith("view_imperfect_"):
            raise ProductionDatasetError(f"Unknown bridge view token: {tokens[2]}")
        cards = []
        hands = {}
        for token in hand_tokens:
            _, seat, card = token.split(":", 2)
            if seat not in {"N", "E", "S", "W"}:
                raise ProductionDatasetError(f"Invalid bridge hand seat: {token}")
            hands.setdefault(seat, []).append(card)
            cards.append(card)
        if tokens[2] == "view_omniscient" and (len(cards) != 52 or len(set(cards)) != 52):
            raise ProductionDatasetError("Bridge entry must contain 52 unique dealt cards")
        if any(len(hand_cards) != 13 for hand_cards in hands.values()):
            raise ProductionDatasetError("Bridge hand-card tokens must group into 13 cards per seat")
        for card in cards:
            if not re.fullmatch(r"[AKQJT98765432][shdc]", card):
                raise ProductionDatasetError(f"Invalid bridge card: {card}")
        dealer = None
        play_leader = None
        trump_suit = None
        calls = []
        played_cards = []
        played_seats = []
        for token in tokens[3:-1]:
            if token.startswith("dealer:"):
                dealer = token.split(":", 1)[1]
                continue
            if token.startswith("play_leader:"):
                play_leader = token.split(":", 1)[1]
                continue
            if token.startswith("trump:"):
                trump_suit = token.split(":", 1)[1]
                if trump_suit not in {"c", "d", "h", "s"}:
                    raise ProductionDatasetError(f"Invalid bridge trump token: {token}")
                continue
            if token.startswith(("vul:", "contract:", "declarer:", "hand:")):
                continue
            if token.startswith("bid:"):
                call = token.split(":", 1)[1]
                if not re.fullmatch(r"(?:PASS|X|XX|[1-7][CDHSN])", call):
                    raise ProductionDatasetError(f"Invalid bridge bid token: {token}")
                calls.append(call)
                continue
            if token.startswith("play:"):
                parts = token.split(":")
                if len(parts) != 3:
                    raise ProductionDatasetError(f"Invalid bridge play token: {token}")
                _, seat, card = parts
                if seat not in {"N", "E", "S", "W"}:
                    raise ProductionDatasetError(f"Invalid bridge play seat: {token}")
                if not re.fullmatch(r"[AKQJT98765432][shdc]", card):
                    raise ProductionDatasetError(f"Invalid bridge play token: {token}")
                played_seats.append(seat)
                played_cards.append(card)
                continue
            raise ProductionDatasetError(f"Invalid bridge token: {token}")
        try:
            validate_bridge_auction(calls, dealer, require_terminated=bool(played_cards))
            if played_cards and play_leader:
                expected_seats = bridge_expected_play_seats(played_cards, play_leader, trump_suit)
                if played_seats != expected_seats:
                    raise ProductionDatasetError("Bridge play seat annotations do not match trick order")
            if tokens[2] == "view_omniscient" and played_cards:
                validate_bridge_play(played_cards, hands, play_leader, trump_suit)
            elif tokens[2].startswith("view_imperfect_") and played_cards and hand_tokens:
                visible_hands = {}
                for token in hand_tokens:
                    _, seat, card = token.split(":")
                    visible_hands.setdefault(seat, set()).add(card)
                visible_remaining = {seat: set(cards) for seat, cards in visible_hands.items()}
                for seat, card in zip(played_seats, played_cards):
                    if seat in visible_remaining:
                        if card not in visible_remaining[seat]:
                            raise ProductionDatasetError(f"Bridge visible hand cannot play {card}")
                        visible_remaining[seat].remove(card)
        except Exception as exc:
            raise ProductionDatasetError(f"Invalid bridge sequence: {exc}") from exc


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

    def write(self, entry, starts_new_view_group=True):
        validate_entry(entry)
        row = self.row_transform(entry) if self.row_transform else entry
        if self.current_file is None:
            self._open_next()

        token_count = row_token_count(row)
        if starts_new_view_group and self.current_rows > 0 and self.current_tokens + token_count > self.max_tokens_per_shard:
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
    cached_entries_path=None,
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
        if cached_entries_path:
            entry_iter = limit_entries(iter_cached_entries(cached_entries_path), game, max_records=max_records)
        else:
            entry_iter = iter_game_entries(game, input_paths, max_records=max_records)
        for entry in entry_iter:
            metadata = entry.get("metadata") or {}
            starts_new_view_group = game not in {"poker", "bridge"} or metadata.get("view_type") == "complete"
            if total_tokens >= target_tokens and starts_new_view_group:
                break
            stats_entry = entry
            if output_format == "mahjonglm_jsonl":
                stats_entry = {**entry, "tokens": entry_to_mahjonglm_row_tokens(entry)}
            completed = writer.write(entry, starts_new_view_group=starts_new_view_group)
            stats.update(stats_entry)
            total_tokens += row_token_count(stats_entry)
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
        "output_format": output_format,
        "tokenizer_fingerprint": tokenizer.fingerprint() if tokenizer is not None else None,
        "target_tokens": target_tokens,
        "token_deficit": max(target_tokens - total_tokens, 0),
        "shards": writer.completed,
        "uploaded": uploaded,
        "stats": stats.summary(target_tokens_per_game=target_tokens).get(game, {}),
    }


def load_source_catalog(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def source_catalog_entry(catalog, game, source_name):
    for source in catalog.get("games", {}).get(game, []):
        if source.get("name") == source_name:
            return source
    raise ProductionDatasetError(f"Source '{source_name}' is not listed for {game} in source catalog")


def assert_source_allowed_for_primary_build(catalog, game, source_name, allow_fallback=False):
    source = source_catalog_entry(catalog, game, source_name)
    source_class = source.get("source_class")
    quality_tier = source.get("quality_tier")
    primary_tiers = {"primary_3b", "primary_3b_generated", "primary_or_mix"}
    if source_class in {"engine_top", "human_top"} and quality_tier in primary_tiers:
        return source
    if allow_fallback and quality_tier in {"filtered_fallback_only", "primary_or_mix", "candidate_primary"}:
        return source
    raise ProductionDatasetError(
        f"Source '{source_name}' is not allowed for a primary {game} build "
        f"(source_class={source_class}, quality_tier={quality_tier})"
    )


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
