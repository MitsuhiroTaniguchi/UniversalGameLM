import hashlib
import re


DEFAULT_SEAT_COUNTS = {
    "mahjong": 4,
    "chess": 2,
    "shogi": 2,
    "go": 2,
    "othello": 2,
    "bridge": 4,
}


def _stable_game_id(entry):
    metadata = entry.get("metadata") or {}
    if metadata.get("view_group_id"):
        raw = str(metadata["view_group_id"])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    raw = "|".join([
        str(entry.get("game", "")),
        str(metadata.get("source_id", "")),
        str(metadata.get("source_name", "")),
        str(metadata.get("filename", "")),
        str(metadata.get("hand_index", "")),
        " ".join(entry.get("tokens") or []),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _year_from_metadata(metadata):
    for key in ("year", "date"):
        value = metadata.get(key)
        if value is None:
            continue
        match = re.search(r"\d{4}", str(value))
        if match:
            return int(match.group(0))
    return None


def tokens_to_mahjonglm_stream(entry):
    """Converts internal BOS/game/EOS tokens to MahjongLM-style body tokens.

    Strips BOS, game marker, and EOS.  Ensures the body starts with a
    ``view:`` token — if one is not already present, ``view:complete``
    is prepended.
    """
    game = entry["game"]
    tokens = list(entry["tokens"])
    if len(tokens) < 4 or tokens[0] != "<bos>" or tokens[-1] != "<eos>":
        raise ValueError("Internal entry must include BOS/EOS before conversion")
    marker = f"<{game}>"
    if tokens[1] != marker:
        raise ValueError(f"Expected {marker}, got {tokens[1]}")

    body = tokens[2:-1]
    if body and body[0].startswith("view:"):
        return body
    return ["view:complete"] + body


def normalize_mahjonglm_metadata(entry):
    game = entry["game"]
    metadata = entry.get("metadata") or {}
    view_type = metadata.get("view_type") or "complete"
    viewer_seat = metadata.get("viewer_seat")
    seat_count = metadata.get("seat_count")
    if seat_count is None:
        seat_count = DEFAULT_SEAT_COUNTS.get(game)
    if seat_count is None:
        if game == "poker":
            raise ValueError("Missing seat_count for poker; poker table size is variable, so metadata.seat_count is required")
        raise ValueError(f"Missing seat_count for {game}")

    return {
        "game_id": metadata.get("game_id") or _stable_game_id(entry),
        "year": _year_from_metadata(metadata),
        "seat_count": int(seat_count),
        "view_type": view_type,
        "viewer_seat": viewer_seat,
    }


def entry_to_mahjonglm_row(entry, tokenizer):
    stream_tokens = tokens_to_mahjonglm_stream(entry)
    ids = tokenizer.encode_strict(stream_tokens)
    row = normalize_mahjonglm_metadata(entry)
    row["length"] = len(ids)
    row["input_ids"] = ids
    row["tokenizer_fingerprint"] = tokenizer.fingerprint()
    return row


def collect_tokens_for_mahjonglm(entries):
    for entry in entries:
        yield from tokens_to_mahjonglm_stream(entry)


def entry_to_mahjonglm_row_tokens(entry):
    return tokens_to_mahjonglm_stream(entry)


# ---------------------------------------------------------------------------
# MahjongLM → Unified dataset ID conversion
# ---------------------------------------------------------------------------

def build_mahjonglm_to_universal_id_map(
    old_vocab_or_dir=None,
    universal_vocab_path=None,
):
    """Build ``{old_mahjonglm_id: new_universal_id}`` mapping.

    *old_vocab_or_dir* can be:
    - a ``dict`` (old token → id mapping)
    - a ``str`` path to a MahjongLM tokenizer directory
    - ``None`` (uses the canonical 828-token old vocab)
    """
    from src.vocab_migration import build_mahjonglm_to_universal_mapping
    return build_mahjonglm_to_universal_mapping(
        old_vocab_or_dir if isinstance(old_vocab_or_dir, str) else None,
        universal_vocab_path,
    )


def convert_mahjonglm_ids(
    ids,
    mapping=None,
    old_vocab_or_dir=None,
    universal_vocab_path=None,
):
    """Convert a list of MahjongLM IDs to unified vocab IDs."""
    from src.vocab_migration import convert_ids, build_mahjonglm_to_universal_mapping
    if mapping is None:
        mapping = build_mahjonglm_to_universal_mapping(old_vocab_or_dir, universal_vocab_path)
    return convert_ids(ids, mapping)


def convert_mahjonglm_dataset_row(
    row,
    mapping=None,
    old_vocab_or_dir=None,
    universal_vocab_path=None,
):
    """Convert a MahjongLM dataset row's ``input_ids`` to unified vocab."""
    from src.vocab_migration import convert_mahjonglm_row as _convert_row
    if mapping is None:
        from src.vocab_migration import build_mahjonglm_to_universal_mapping
        mapping = build_mahjonglm_to_universal_mapping(old_vocab_or_dir, universal_vocab_path)
    return _convert_row(row, mapping)
