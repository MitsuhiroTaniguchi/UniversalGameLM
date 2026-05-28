"""Vocabulary migration from MahjongLM 828-token legacy names to new names.

Maps every old token name to its new name while preserving integer IDs.
The rename rules convert 0-indexed seat numbers to wind directions and
restructure token names to use colon-delimited prefixes.
"""

import re

SEAT_TO_WIND = {0: "E", 1: "S", 2: "W", 3: "N"}

# Tiles in canonical order (37 tiles: m1-m9,m0, p1-p9,p0, s1-s9,s0, z1-z7)
TILES = (
    [f"m{i}" for i in range(1, 10)] + ["m0"]
    + [f"p{i}" for i in range(1, 10)] + ["p0"]
    + [f"s{i}" for i in range(1, 10)] + ["s0"]
    + [f"z{i}" for i in range(1, 8)]
)

# TENBO numeric values
TENBO_NUMERICS = [100, 200, 300, 400, 500, 600, 700, 800, 900,
                  1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

# Self-turn action types
SELF_ACTION_TYPES = ["ankan", "kakan", "riichi", "tsumo", "kyushukyuhai"]

# React action types
REACT_ACTION_TYPES = ["chi", "pon", "daiminkan", "ron"]

# Yaku names (in order)
YAKU_NAMES = [
    "menzen_tsumo", "riichi", "ippatsu", "chankan", "rinshan_kaihou",
    "haitei_raoyue", "houtei_raoyui", "pinfu", "tanyao", "iipeiko",
    "jikaze_ton", "jikaze_nan", "jikaze_sha", "jikaze_pei",
    "bakaze_ton", "bakaze_nan", "bakaze_sha", "bakaze_pei",
    "sangenpai_haku", "sangenpai_hatsu", "sangenpai_chun",
    "daburu_riichi", "chiitoitsu", "chanta", "ittsu",
    "sanshoku_doujun", "sanshoku_doukou", "sankantsu", "toitoi",
    "sanankou", "shousangen", "honroutou", "ryanpeiko",
    "junchan", "honitsu", "chinitsu",
    "tenhou", "chiihou",
    "daisangen", "suuankou", "suuankou_tanki", "tsuiisou",
    "ryuiisou", "chinroutou", "chuurenpoutou",
    "chuurenpoutou_junsei", "kokushi", "kokushi_13men",
    "daisuushii", "shousuushii", "sukantsu",
    "dora", "uradora", "akadora",
]

# Fu values
FU_VALUES = [20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]

# Pingju types
PINGJU_TYPES = [
    "ryukyoku", "kyushukyuhai", "nagashimangan",
    "sufurenda", "sukantsu", "suuchariichi", "sanchahou",
]


def rename_token(old_name: str) -> str:
    """Convert an old MahjongLM token name to its new name.

    Returns the new name, or the old name unchanged for tokens that
    are not renamed (specials, boundaries).
    """
    # --- Special tokens (unchanged) ---
    if old_name in ("<pad>", "<unk>", "<bos>", "<eos>",
                     "game_start", "game_end"):
        return old_name

    # --- Boundary tokens (unchanged) ---
    if old_name in ("round_start", "round_end"):
        return old_name

    # --- View tokens ---
    if old_name == "view_complete":
        return "view:complete"
    if old_name == "view_omniscient":
        return "view:omniscient"
    m = re.fullmatch(r"view_imperfect_(\d)", old_name)
    if m:
        return f"view:imperfect:p{int(m.group(1)) + 1}"

    # --- Rule tokens ---
    m = re.fullmatch(r"rule_player_(\d)", old_name)
    if m:
        return f"mj:rule:player:{m.group(1)}"
    m = re.fullmatch(r"rule_length_(\w+)", old_name)
    if m:
        return f"mj:rule:length:{m.group(1)}"

    # --- Wall ---
    if old_name == "wall":
        return "mj:wall"

    # --- Bakaze ---
    m = re.fullmatch(r"bakaze_(\d)", old_name)
    if m:
        return f"mj:bakaze:{SEAT_TO_WIND[int(m.group(1))]}"

    # --- Kyoku (1-indexed) ---
    m = re.fullmatch(r"kyoku_(\d)", old_name)
    if m:
        return f"mj:kyoku:{int(m.group(1)) + 1}"

    # --- Honba / riichi_sticks ---
    if old_name == "honba":
        return "mj:honba"
    if old_name == "riichi_sticks":
        return "mj:riichi_sticks"

    # --- TENBO ---
    if old_name == "TENBO_ZERO":
        return "mj:tenbo:zero"
    if old_name == "TENBO_PLUS":
        return "mj:tenbo:plus"
    if old_name == "TENBO_MINUS":
        return "mj:tenbo:minus"
    m = re.fullmatch(r"TENBO_(\d+)", old_name)
    if m:
        return f"mj:tenbo:{m.group(1)}"

    # --- Dora ---
    if old_name == "dora":
        return "mj:dora"
    if old_name == "ura_dora":
        return "mj:ura_dora"

    # --- Haipai (seat -> wind) ---
    m = re.fullmatch(r"hidden_haipai_(\d)", old_name)
    if m:
        return f"mj:hidden_haipai:{SEAT_TO_WIND[int(m.group(1))]}"
    m = re.fullmatch(r"haipai_(\d)", old_name)
    if m:
        return f"mj:haipai:{SEAT_TO_WIND[int(m.group(1))]}"

    # --- Tiles (bare tile names -> mj:tile) ---
    if old_name in TILES:
        return f"mj:{old_name}"

    # --- Draw (draw_{seat}_{tile_or_hidden}) ---
    m = re.fullmatch(r"draw_(\d)_(.+)", old_name)
    if m:
        seat = int(m.group(1))
        rest = m.group(2)
        return f"mj:draw:{SEAT_TO_WIND[seat]}:{rest}"

    # --- Discard (discard_{seat}_{tile}_{type}) ---
    m = re.fullmatch(r"discard_(\d)_(.+?)_(tedashi|tsumogiri)", old_name)
    if m:
        seat = int(m.group(1))
        tile = m.group(2)
        dtype = m.group(3)
        return f"mj:discard:{SEAT_TO_WIND[seat]}:{tile}:{dtype}"

    # --- Decision tokens: pass_react with reason ---
    # Must check before generic react pattern since it has an extra suffix
    m = re.fullmatch(r"pass_react_(\d)_(\w+?)_(forced_priority|voluntary)", old_name)
    if m:
        seat = int(m.group(1))
        action_type = m.group(2)
        reason = m.group(3)
        return f"mj:pass:react:{SEAT_TO_WIND[seat]}:{action_type}:{reason}"

    # --- Decision tokens: self actions ---
    m = re.fullmatch(r"(opt|take|pass)_self_(\d)_(\w+)", old_name)
    if m:
        action = m.group(1)
        seat = int(m.group(2))
        action_type = m.group(3)
        return f"mj:{action}:self:{SEAT_TO_WIND[seat]}:{action_type}"

    # --- Decision tokens: react actions (opt, take, pass without reason) ---
    m = re.fullmatch(r"(opt|take|pass)_react_(\d)_(\w+)", old_name)
    if m:
        action = m.group(1)
        seat = int(m.group(2))
        action_type = m.group(3)
        return f"mj:{action}:react:{SEAT_TO_WIND[seat]}:{action_type}"

    # --- Chi position / red ---
    if old_name == "chi_pos_low":
        return "mj:chi_pos:low"
    if old_name == "chi_pos_mid":
        return "mj:chi_pos:mid"
    if old_name == "chi_pos_high":
        return "mj:chi_pos:high"
    if old_name == "red_used":
        return "mj:red:used"
    if old_name == "red_not_used":
        return "mj:red:not_used"

    # --- Hule ---
    m = re.fullmatch(r"hule_(\d)", old_name)
    if m:
        return f"mj:hule:{SEAT_TO_WIND[int(m.group(1))]}"

    # --- Opened hand ---
    m = re.fullmatch(r"opened_hand_(\d)", old_name)
    if m:
        return f"mj:opened_hand:{SEAT_TO_WIND[int(m.group(1))]}"

    # --- Score tokens ---
    # final_score uses 1-indexed player number (p1 = oya, p2 = shimocha, ...)
    m = re.fullmatch(r"final_score_(\d)", old_name)
    if m:
        return f"mj:final_score:p{int(m.group(1)) + 1}"
    m = re.fullmatch(r"score_delta_(\d)", old_name)
    if m:
        return f"mj:score_delta:{SEAT_TO_WIND[int(m.group(1))]}"
    m = re.fullmatch(r"score_(\d)", old_name)
    if m:
        return f"mj:score:{SEAT_TO_WIND[int(m.group(1))]}"

    # --- Rank tokens ---
    # final_rank uses 1-indexed player number
    m = re.fullmatch(r"final_rank_(\d)_(\d)", old_name)
    if m:
        seat = int(m.group(1))
        rank = m.group(2)
        return f"mj:final_rank:p{seat + 1}:{rank}"
    m = re.fullmatch(r"rank_(\d)_(\d)", old_name)
    if m:
        seat = int(m.group(1))
        rank = m.group(2)
        return f"mj:rank:{SEAT_TO_WIND[seat]}:{rank}"

    # --- Yaku ---
    m = re.fullmatch(r"yaku_(\w+)", old_name)
    if m:
        return f"mj:yaku:{m.group(1)}"

    # --- Han ---
    m = re.fullmatch(r"han_(\d+)", old_name)
    if m:
        return f"mj:han:{m.group(1)}"

    # --- Fu ---
    m = re.fullmatch(r"fu_(\d+)", old_name)
    if m:
        return f"mj:fu:{m.group(1)}"

    # --- Yakuman ---
    m = re.fullmatch(r"yakuman_(\d+)", old_name)
    if m:
        return f"mj:yakuman:{m.group(1)}"

    # --- Pingju ---
    m = re.fullmatch(r"pingju_(\w+)", old_name)
    if m:
        return f"mj:pingju:{m.group(1)}"

    # Fallback: return unchanged (should not happen for valid vocab)
    return old_name


def build_rename_map(old_vocab: dict) -> dict:
    """Return ``{old_name: new_name}`` for every token in *old_vocab*.

    *old_vocab* maps ``token_name -> int_id``.
    """
    return {name: rename_token(name) for name in old_vocab}


def migrate_vocab(old_vocab: dict) -> dict:
    """Return ``{new_name: old_id}`` preserving all original integer IDs."""
    rename_map = build_rename_map(old_vocab)
    return {rename_map[name]: idx for name, idx in old_vocab.items()}


def validate_migration(old_vocab: dict, new_vocab: dict) -> None:
    """Check that *new_vocab* is a valid migration of *old_vocab*.

    Raises ``ValueError`` with a descriptive message on any failure.
    """
    errors = []

    # 1. All 828 IDs are preserved
    old_ids = set(old_vocab.values())
    new_ids = set(new_vocab.values())
    missing_ids = old_ids - new_ids
    extra_ids = new_ids - old_ids
    if missing_ids:
        errors.append(f"Missing IDs in new vocab: {sorted(missing_ids)[:20]}")
    if extra_ids:
        errors.append(f"Extra IDs in new vocab: {sorted(extra_ids)[:20]}")
    if len(old_vocab) != len(new_vocab):
        errors.append(
            f"Size mismatch: old={len(old_vocab)}, new={len(new_vocab)}"
        )

    # 2. No duplicate new names
    if len(new_vocab) != len(set(new_vocab.keys())):
        from collections import Counter
        counts = Counter(new_vocab.keys())
        dupes = {k: v for k, v in counts.items() if v > 1}
        errors.append(f"Duplicate new names: {dupes}")

    # 3. All old tokens have a mapping (every old ID maps to exactly one new name)
    rename_map = build_rename_map(old_vocab)
    unmapped = [name for name in old_vocab if rename_map.get(name) == name
                and name not in ("<pad>", "<unk>", "<bos>", "<eos>",
                                 "game_start", "game_end",
                                 "round_start", "round_end")]
    if unmapped:
        errors.append(f"Tokens returned unchanged (possibly unmapped): {unmapped[:20]}")

    if errors:
        raise ValueError("Migration validation failed:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# Pre-computed RENAME_MAP for the canonical 828-token vocabulary
# ---------------------------------------------------------------------------

def _build_canonical_old_vocab() -> dict:
    """Reconstruct the canonical 828-token old vocabulary with correct IDs."""
    vocab = {}
    idx = 0

    def _add(name):
        nonlocal idx
        vocab[name] = idx
        idx += 1

    # 0-5: Special
    for t in ["<pad>", "<unk>", "<bos>", "<eos>", "game_start", "game_end"]:
        _add(t)

    # 6-11: View
    for t in ["view_complete",
              "view_imperfect_0", "view_imperfect_1",
              "view_imperfect_2", "view_imperfect_3",
              "view_omniscient"]:
        _add(t)

    # 12-15: Rule
    for t in ["rule_player_3", "rule_player_4",
              "rule_length_tonpu", "rule_length_hanchan"]:
        _add(t)

    # 16: round_start
    _add("round_start")

    # 17: wall
    _add("wall")

    # 18: round_end
    _add("round_end")

    # 19-21: bakaze
    for i in range(3):
        _add(f"bakaze_{i}")

    # 22-25: kyoku
    for i in range(4):
        _add(f"kyoku_{i}")

    # 26-27: honba, riichi_sticks
    _add("honba")
    _add("riichi_sticks")

    # 28-49: TENBO (22 tokens)
    _add("TENBO_ZERO")
    _add("TENBO_PLUS")
    _add("TENBO_MINUS")
    for v in TENBO_NUMERICS:
        _add(f"TENBO_{v}")

    # 50-51: dora, ura_dora
    _add("dora")
    _add("ura_dora")

    # 52-59: haipai (hidden then regular, each for seats 0-3)
    for i in range(4):
        _add(f"hidden_haipai_{i}")
    for i in range(4):
        _add(f"haipai_{i}")

    # 60-96: tiles (37 tiles)
    for tile in TILES:
        _add(tile)

    # 97-544: draw and discard
    # draw: 4 seats * (1 hidden + 37 tiles) = 4 * 38 = 152 tokens
    # discard: 4 seats * 37 tiles * 2 types = 296 tokens
    # Total: 152 + 296 = 448 tokens (IDs 97-544)
    for seat in range(4):
        _add(f"draw_{seat}_hidden")
        for tile in TILES:
            _add(f"draw_{seat}_{tile}")
    for seat in range(4):
        for tile in TILES:
            _add(f"discard_{seat}_{tile}_tedashi")
            _add(f"discard_{seat}_{tile}_tsumogiri")

    # 545-676: decision tokens (132 tokens)
    # Self actions: 4 seats * 5 types * 3 (opt/take/pass) = 60
    for seat in range(4):
        for action_type in SELF_ACTION_TYPES:
            _add(f"opt_self_{seat}_{action_type}")
            _add(f"take_self_{seat}_{action_type}")
            _add(f"pass_self_{seat}_{action_type}")

    # React actions: 4 seats * 4 types * (opt + take + pass_fp + pass_vol) = 64
    for seat in range(4):
        for action_type in REACT_ACTION_TYPES:
            _add(f"opt_react_{seat}_{action_type}")
            _add(f"take_react_{seat}_{action_type}")
            _add(f"pass_react_{seat}_{action_type}_forced_priority")
            _add(f"pass_react_{seat}_{action_type}_voluntary")

    # "none" react: blanket skip when no call is available (2 per seat = 8)
    for seat in range(4):
        _add(f"opt_react_{seat}_none")
        _add(f"pass_react_{seat}_none_voluntary")

    # 677-681: chi/red
    for t in ["chi_pos_low", "chi_pos_mid", "chi_pos_high",
              "red_used", "red_not_used"]:
        _add(t)

    # 682-689: hule + opened_hand
    for i in range(4):
        _add(f"hule_{i}")
    for i in range(4):
        _add(f"opened_hand_{i}")

    # 690-701: score tokens
    for i in range(4):
        _add(f"score_{i}")
    for i in range(4):
        _add(f"score_delta_{i}")
    for i in range(4):
        _add(f"final_score_{i}")

    # 702-733: rank tokens (seat 0-3, rank 1-4)
    for i in range(4):
        for r in range(1, 5):
            _add(f"rank_{i}_{r}")
    for i in range(4):
        for r in range(1, 5):
            _add(f"final_rank_{i}_{r}")

    # 734-787: yaku (54 tokens)
    for yaku in YAKU_NAMES:
        _add(f"yaku_{yaku}")

    # 788-800: han (13 tokens)
    for h in range(1, 14):
        _add(f"han_{h}")

    # 801-814: fu (14 tokens)
    for fu in FU_VALUES:
        _add(f"fu_{fu}")

    # 815-820: yakuman (6 tokens)
    for y in range(1, 7):
        _add(f"yakuman_{y}")

    # 821-827: pingju (7 tokens)
    for ptype in PINGJU_TYPES:
        _add(f"pingju_{ptype}")

    assert idx == 828, f"Expected 828 tokens, got {idx}"
    return vocab


_CANONICAL_OLD_VOCAB = _build_canonical_old_vocab()

#: Pre-computed ``{old_name: new_name}`` for the canonical 828-token vocabulary.
RENAME_MAP = build_rename_map(_CANONICAL_OLD_VOCAB)
