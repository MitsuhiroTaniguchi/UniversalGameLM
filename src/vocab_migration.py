"""Vocabulary migration utilities.

Covers:
- Renaming old MahjongLM token names to new colon-delimited names.
- Building old-ID → new-ID mapping for migrating existing mahjonglm-dataset.
- Converting rows between vocab versions.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

SEAT_TO_WIND = {0: "E", 1: "S", 2: "W", 3: "N"}

# Tiles in canonical order (37 tiles: m1-m9,m0, p1-p9,p0, s1-s9,s0, z1-z7)
TILES = (
    [f"m{i}" for i in range(1, 10)] + ["m0"]
    + [f"p{i}" for i in range(1, 10)] + ["p0"]
    + [f"s{i}" for i in range(1, 10)] + ["s0"]
    + [f"z{i}" for i in range(1, 8)]
)

TENBO_NUMERICS = [100, 200, 300, 400, 500, 600, 700, 800, 900,
                  1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

SELF_ACTION_TYPES = ["ankan", "kakan", "riichi", "tsumo", "kyushukyuhai"]
REACT_ACTION_TYPES = ["chi", "pon", "daiminkan", "ron"]

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

FU_VALUES = [20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]

PINGJU_TYPES = [
    "ryukyoku", "kyushukyuhai", "nagashimangan",
    "sufurenda", "sukantsu", "suuchariichi", "sanchahou",
]


def rename_token(old_name: str) -> str:
    """Convert an old MahjongLM token name to its new name."""
    if old_name in ("<pad>", "<unk>", "<bos>", "<eos>",
                     "game_start", "game_end"):
        return old_name
    if old_name in ("round_start", "round_end"):
        return old_name
    if old_name == "view_complete":
        return "view:complete"
    if old_name == "view_omniscient":
        return "view:omniscient"
    m = re.fullmatch(r"view_imperfect_(\d)", old_name)
    if m:
        return f"view:imperfect:p{int(m.group(1)) + 1}"
    m = re.fullmatch(r"rule_player_(\d)", old_name)
    if m:
        return f"mj:rule:player:{m.group(1)}"
    m = re.fullmatch(r"rule_length_(\w+)", old_name)
    if m:
        return f"mj:rule:length:{m.group(1)}"
    if old_name == "wall":
        return "mj:wall"
    m = re.fullmatch(r"bakaze_(\d)", old_name)
    if m:
        return f"mj:bakaze:{SEAT_TO_WIND[int(m.group(1))]}"
    m = re.fullmatch(r"kyoku_(\d)", old_name)
    if m:
        return f"mj:kyoku:{int(m.group(1)) + 1}"
    if old_name == "honba":
        return "mj:honba"
    if old_name == "riichi_sticks":
        return "mj:riichi_sticks"
    if old_name == "TENBO_ZERO":
        return "mj:tenbo:zero"
    if old_name == "TENBO_PLUS":
        return "mj:tenbo:plus"
    if old_name == "TENBO_MINUS":
        return "mj:tenbo:minus"
    m = re.fullmatch(r"TENBO_(\d+)", old_name)
    if m:
        return f"mj:tenbo:{m.group(1)}"
    if old_name == "dora":
        return "mj:dora"
    if old_name == "ura_dora":
        return "mj:ura_dora"
    m = re.fullmatch(r"hidden_haipai_(\d)", old_name)
    if m:
        return f"mj:hidden_haipai:{SEAT_TO_WIND[int(m.group(1))]}"
    m = re.fullmatch(r"haipai_(\d)", old_name)
    if m:
        return f"mj:haipai:{SEAT_TO_WIND[int(m.group(1))]}"
    if old_name in TILES:
        return f"mj:{old_name}"
    m = re.fullmatch(r"draw_(\d)_(.+)", old_name)
    if m:
        seat = int(m.group(1))
        rest = m.group(2)
        return f"mj:draw:{SEAT_TO_WIND[seat]}:{rest}"
    m = re.fullmatch(r"discard_(\d)_(.+?)_(tedashi|tsumogiri)", old_name)
    if m:
        seat = int(m.group(1))
        tile = m.group(2)
        dtype = m.group(3)
        return f"mj:discard:{SEAT_TO_WIND[seat]}:{tile}:{dtype}"
    m = re.fullmatch(r"pass_react_(\d)_(\w+?)_(forced_priority|voluntary)", old_name)
    if m:
        seat = int(m.group(1))
        action_type = m.group(2)
        reason = m.group(3)
        return f"mj:pass:react:{SEAT_TO_WIND[seat]}:{action_type}:{reason}"
    m = re.fullmatch(r"(opt|take|pass)_self_(\d)_(\w+)", old_name)
    if m:
        action = m.group(1)
        seat = int(m.group(2))
        action_type = m.group(3)
        return f"mj:{action}:self:{SEAT_TO_WIND[seat]}:{action_type}"
    m = re.fullmatch(r"(opt|take|pass)_react_(\d)_(\w+)", old_name)
    if m:
        action = m.group(1)
        seat = int(m.group(2))
        action_type = m.group(3)
        return f"mj:{action}:react:{SEAT_TO_WIND[seat]}:{action_type}"
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
    m = re.fullmatch(r"hule_(\d)", old_name)
    if m:
        return f"mj:hule:{SEAT_TO_WIND[int(m.group(1))]}"
    m = re.fullmatch(r"opened_hand_(\d)", old_name)
    if m:
        return f"mj:opened_hand:{SEAT_TO_WIND[int(m.group(1))]}"
    m = re.fullmatch(r"final_score_(\d)", old_name)
    if m:
        return f"mj:final_score:p{int(m.group(1)) + 1}"
    m = re.fullmatch(r"score_delta_(\d)", old_name)
    if m:
        return f"mj:score_delta:{SEAT_TO_WIND[int(m.group(1))]}"
    m = re.fullmatch(r"score_(\d)", old_name)
    if m:
        return f"mj:score:{SEAT_TO_WIND[int(m.group(1))]}"
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
    m = re.fullmatch(r"yaku_(\w+)", old_name)
    if m:
        return f"mj:yaku:{m.group(1)}"
    m = re.fullmatch(r"han_(\d+)", old_name)
    if m:
        return f"mj:han:{m.group(1)}"
    m = re.fullmatch(r"fu_(\d+)", old_name)
    if m:
        return f"mj:fu:{m.group(1)}"
    m = re.fullmatch(r"yakuman_(\d+)", old_name)
    if m:
        return f"mj:yakuman:{m.group(1)}"
    m = re.fullmatch(r"pingju_(\w+)", old_name)
    if m:
        return f"mj:pingju:{m.group(1)}"
    return old_name


def build_rename_map(old_vocab: dict) -> dict:
    return {name: rename_token(name) for name in old_vocab}


def migrate_vocab(old_vocab: dict) -> dict:
    """Return ``{new_name: old_id}`` preserving all original integer IDs."""
    rename_map = build_rename_map(old_vocab)
    return {rename_map[name]: idx for name, idx in old_vocab.items()}


# ---------------------------------------------------------------------------
# Unified vocab loading
# ---------------------------------------------------------------------------

def load_unified_vocab(vocab_path: Optional[str] = None) -> Dict[str, int]:
    """Load the unified vocabulary as ``{token: id}``.

    If *vocab_path* is ``None``, looks for ``vocab/universal.json``
    relative to the repo root.
    """
    if vocab_path is None:
        vocab_path = str(Path(__file__).parent.parent / "vocab" / "universal.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_unified_vocab_tokens(vocab_path: Optional[str] = None) -> List[str]:
    """Load the unified vocabulary as an ordered list of tokens."""
    if vocab_path is None:
        vocab_path = str(Path(__file__).parent.parent / "vocab" / "universal.txt")
    tokens = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.rstrip("\n")
            if token:
                tokens.append(token)
    return tokens


# ---------------------------------------------------------------------------
# ID mapping between vocabularies
# ---------------------------------------------------------------------------

def build_id_mapping(
    old_vocab: Dict[str, int],
    new_vocab: Dict[str, int],
) -> Dict[int, int]:
    """Build ``{old_id: new_id}`` mapping by matching token names.

    Every token in *old_vocab* is looked up in *new_vocab* by name;
    if a name is not found, the old name is passed through
    :func:`rename_token` first before lookup.
    """
    mapping = {}
    missing = []
    for old_token, old_id in old_vocab.items():
        if old_token in new_vocab:
            mapping[old_id] = new_vocab[old_token]
        else:
            renamed = rename_token(old_token)
            if renamed in new_vocab:
                mapping[old_id] = new_vocab[renamed]
            else:
                missing.append(old_token)
    if missing:
        from warnings import warn
        warn(f"vocab_migration: {len(missing)} token(s) had no mapping: {missing[:20]}")
        for token in missing:
            mapping[old_vocab[token]] = new_vocab.get("<unk>", 1)
    return mapping


def convert_ids(ids: List[int], mapping: Dict[int, int], unk_id: int = 1) -> List[int]:
    """Apply an ID mapping to a list of token IDs.

    IDs not found in *mapping* are replaced with *unk_id*.
    """
    return [mapping.get(i, unk_id) for i in ids]


# ---------------------------------------------------------------------------
# MahjongLM dataset conversion
# ---------------------------------------------------------------------------

def _build_canonical_old_vocab() -> dict:
    """Reconstruct the canonical 828-token old vocabulary with correct IDs."""
    vocab = {}
    idx = 0

    def _add(name):
        nonlocal idx
        vocab[name] = idx
        idx += 1

    for t in ["<pad>", "<unk>", "<bos>", "<eos>", "game_start", "game_end"]:
        _add(t)
    for t in ["view_complete",
              "view_imperfect_0", "view_imperfect_1",
              "view_imperfect_2", "view_imperfect_3",
              "view_omniscient"]:
        _add(t)
    for t in ["rule_player_3", "rule_player_4",
              "rule_length_tonpu", "rule_length_hanchan"]:
        _add(t)
    _add("round_start")
    _add("wall")
    _add("round_end")
    for i in range(3):
        _add(f"bakaze_{i}")
    for i in range(4):
        _add(f"kyoku_{i}")
    _add("honba")
    _add("riichi_sticks")
    _add("TENBO_ZERO")
    _add("TENBO_PLUS")
    _add("TENBO_MINUS")
    for v in TENBO_NUMERICS:
        _add(f"TENBO_{v}")
    _add("dora")
    _add("ura_dora")
    for i in range(4):
        _add(f"hidden_haipai_{i}")
    for i in range(4):
        _add(f"haipai_{i}")
    for tile in TILES:
        _add(tile)
    for seat in range(4):
        _add(f"draw_{seat}_hidden")
        for tile in TILES:
            _add(f"draw_{seat}_{tile}")
    for seat in range(4):
        for tile in TILES:
            _add(f"discard_{seat}_{tile}_tedashi")
            _add(f"discard_{seat}_{tile}_tsumogiri")
    for seat in range(4):
        for action_type in SELF_ACTION_TYPES:
            _add(f"opt_self_{seat}_{action_type}")
            _add(f"take_self_{seat}_{action_type}")
            _add(f"pass_self_{seat}_{action_type}")
    for seat in range(4):
        for action_type in REACT_ACTION_TYPES:
            _add(f"opt_react_{seat}_{action_type}")
            _add(f"take_react_{seat}_{action_type}")
            _add(f"pass_react_{seat}_{action_type}_forced_priority")
            _add(f"pass_react_{seat}_{action_type}_voluntary")
    for seat in range(4):
        _add(f"opt_react_{seat}_none")
        _add(f"pass_react_{seat}_none_voluntary")
    for t in ["chi_pos_low", "chi_pos_mid", "chi_pos_high",
              "red_used", "red_not_used"]:
        _add(t)
    for i in range(4):
        _add(f"hule_{i}")
    for i in range(4):
        _add(f"opened_hand_{i}")
    for i in range(4):
        _add(f"score_{i}")
    for i in range(4):
        _add(f"score_delta_{i}")
    for i in range(4):
        _add(f"final_score_{i}")
    for i in range(4):
        for r in range(1, 5):
            _add(f"rank_{i}_{r}")
    for i in range(4):
        for r in range(1, 5):
            _add(f"final_rank_{i}_{r}")
    for yaku in YAKU_NAMES:
        _add(f"yaku_{yaku}")
    for h in range(1, 14):
        _add(f"han_{h}")
    for fu in FU_VALUES:
        _add(f"fu_{fu}")
    for y in range(1, 7):
        _add(f"yakuman_{y}")
    for ptype in PINGJU_TYPES:
        _add(f"pingju_{ptype}")

    assert idx == 828, f"Expected 828 tokens, got {idx}"
    return vocab


def load_mahjonglm_vocab(tokenizer_dir: str) -> Dict[str, int]:
    """Load a MahjongLM tokenizer's vocabulary from its directory.

    Supports ``vocab.txt``, ``vocab.json``, or ``tokenizer.json``.
    """
    tokenizer_dir = Path(tokenizer_dir)
    for candidate in ["vocab.txt", "vocab.json", "tokenizer.json"]:
        path = tokenizer_dir / candidate
        if path.exists():
            break
    else:
        return _build_canonical_old_vocab()

    if candidate == "vocab.txt":
        vocab = {}
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.rstrip("\n")
                if token:
                    vocab[token] = idx
        return vocab
    elif candidate == "vocab.json":
        with open(path, "r", encoding="utf-8") as f:
            return {token: int(idx) for token, idx in json.load(f).items()}
    else:
        from tokenizers import Tokenizer as HFTokenizer
        tk = HFTokenizer.from_file(str(path))
        return tk.get_vocab()


def build_mahjonglm_to_universal_mapping(
    mahjonglm_tokenizer_dir: Optional[str] = None,
    universal_vocab_path: Optional[str] = None,
) -> Dict[int, int]:
    """Build ``{old_mahjonglm_id: new_universal_id}`` mapping.

    If *mahjonglm_tokenizer_dir* is ``None``, the canonical 828-token
    vocab is used.
    """
    if mahjonglm_tokenizer_dir:
        old_vocab = load_mahjonglm_vocab(mahjonglm_tokenizer_dir)
    else:
        old_vocab = _build_canonical_old_vocab()
    new_vocab = load_unified_vocab(universal_vocab_path)
    return build_id_mapping(old_vocab, new_vocab)


def convert_mahjonglm_row(
    row: dict,
    mapping: Dict[int, int],
) -> dict:
    """Convert a single MahjongLM dataset row's ``input_ids`` to unified vocab IDs.

    Returns a new dict with updated ``input_ids`` and ``tokenizer_fingerprint``
    removed (since it no longer applies).
    """
    result = dict(row)
    result["input_ids"] = convert_ids(row.get("input_ids", []), mapping)
    result.pop("tokenizer_fingerprint", None)
    result.pop("_fingerprint", None)
    return result


def validate_migration(old_vocab: dict, new_vocab: dict) -> None:
    errors = []
    old_ids = set(old_vocab.values())
    new_ids = set(new_vocab.values())
    missing_ids = old_ids - new_ids
    extra_ids = new_ids - old_ids
    if missing_ids:
        errors.append(f"Missing IDs in new vocab: {sorted(missing_ids)[:20]}")
    if extra_ids:
        errors.append(f"Extra IDs in new vocab: {sorted(extra_ids)[:20]}")
    if len(old_vocab) != len(new_vocab):
        errors.append(f"Size mismatch: old={len(old_vocab)}, new={len(new_vocab)}")

    if len(new_vocab) != len(set(new_vocab.keys())):
        from collections import Counter
        counts = Counter(new_vocab.keys())
        dupes = {k: v for k, v in counts.items() if v > 1}
        errors.append(f"Duplicate new names: {dupes}")

    rename_map = build_rename_map(old_vocab)
    unmapped = [name for name in old_vocab if rename_map.get(name) == name
                and name not in ("<pad>", "<unk>", "<bos>", "<eos>",
                                 "game_start", "game_end",
                                 "round_start", "round_end")]
    if unmapped:
        errors.append(f"Tokens returned unchanged (possibly unmapped): {unmapped[:20]}")

    if errors:
        raise ValueError("Migration validation failed:\n  " + "\n  ".join(errors))


_CANONICAL_OLD_VOCAB = _build_canonical_old_vocab()
RENAME_MAP = build_rename_map(_CANONICAL_OLD_VOCAB)
