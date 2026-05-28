"""Build unified vocabulary that merges all game vocabs with natural ordering.

Ordering:
  1. Special tokens (<pad>, <unk>, <bos>, <eos>, game marker tokens)
  2. Shared tokens without game prefix (game_start, game_end, view:*)
  3. Mahjong tokens (mj:*)
  4. Chess tokens (ch:*)
  5. Shogi tokens (sh:*)
  6. Go tokens (go:*)
  7. Othello tokens (ot:*)
  8. Poker tokens (pk:*)
  9. Bridge tokens (br:*)

Produces vocab/universal.txt and vocab/universal.json.
"""
import json
from pathlib import Path


VOCAB_DIR = Path(__file__).parent / "vocab"

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<mahjong>",
    "<chess>",
    "<shogi>",
    "<go>",
    "<othello>",
    "<poker>",
    "<bridge>",
]

GAME_ORDER = ["mahjong", "chess", "shogi", "go", "othello", "poker", "bridge"]
GAME_PREFIXES = {"mj", "ch", "sh", "go", "ot", "pk", "br"}


def _has_game_prefix(token):
    """Check if a token has a colon-delimited game prefix (mj:*, ch:*, etc.)."""
    prefix = token.split(":")[0] if ":" in token else ""
    return prefix in GAME_PREFIXES


def load_vocab_tokens(name):
    path = VOCAB_DIR / f"{name}.txt"
    tokens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.rstrip("\n")
            if token:
                tokens.append(token)
    return tokens


def _shared_sort_key(token):
    """Sort key for shared (prefix-free) tokens.

    Groups: game_* first, round_* second, view:* third (contiguous).
    Within each group, logical start-before-end ordering.
    """
    if token.startswith("game_"):
        order = 0 if token == "game_start" else 1
        return (0, order, token)
    if token.startswith("round_"):
        order = 0 if token == "round_start" else 1
        return (1, order, token)
    if token.startswith("view:"):
        parts = token.split(":")
        if len(parts) == 2:
            type_order = {"complete": 0, "omniscient": 99}
            return (2, type_order.get(parts[1], 50), "")
        if len(parts) == 3 and parts[1] == "imperfect":
            seat_str = parts[2].lstrip("p")
            try:
                seat_num = int(seat_str)
            except ValueError:
                seat_num = 99
            return (2, 10, seat_num)
    return (9, 0, token)


def build_unified_tokens():
    seen = set()
    unified = []

    # 1. Special tokens
    for token in SPECIAL_TOKENS:
        if token not in seen:
            seen.add(token)
            unified.append(token)

    # 2. Shared tokens (no game prefix) — collect all, then sort
    shared = []
    for game in GAME_ORDER:
        tokens = load_vocab_tokens(game)
        for token in tokens:
            if token in seen:
                continue
            if not _has_game_prefix(token):
                seen.add(token)
                shared.append(token)
    shared.sort(key=_shared_sort_key)
    unified.extend(shared)

    # 3. Game-specific tokens, grouped by game
    for game in GAME_ORDER:
        tokens = load_vocab_tokens(game)
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unified.append(token)

    return unified


def build_unified_vocab():
    unified_tokens = build_unified_tokens()
    vocab = {token: idx for idx, token in enumerate(unified_tokens)}
    return vocab


def main():
    vocab = build_unified_vocab()

    output_txt = VOCAB_DIR / "universal.txt"
    output_json = VOCAB_DIR / "universal.json"

    with open(output_txt, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print(f"Unified vocabulary: {len(vocab)} tokens")
    print(f"  -> {output_txt}")
    print(f"  -> {output_json}")

    # Report breakdown
    counts = {}
    for token in vocab:
        if token.startswith("<") and token.endswith(">"):
            prefix = "special"
        elif token.startswith("view:"):
            prefix = "view"
        elif token in ("game_start", "game_end"):
            prefix = "meta"
        elif _has_game_prefix(token):
            prefix = token.split(":")[0]
        else:
            prefix = "other"
        counts[prefix] = counts.get(prefix, 0) + 1
    for key in sorted(counts):
        print(f"    {key}: {counts[key]}")


if __name__ == "__main__":
    main()
