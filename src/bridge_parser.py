import os
import re
from pathlib import Path


SEATS = ("N", "E", "S", "W")
SUITS = ("S", "H", "D", "C")
RANKS = "AKQJT98765432"
ALL_CARDS = {f"{s}{r}" for s in SUITS for r in RANKS}
CALL_RE = re.compile(r"^(?:PASS|P|X|XX|DBL|RDBL|[1-7](?:C|D|H|S|N|NT))$", re.IGNORECASE)
CARD_RE = re.compile(r"^[SHDC][AKQJT98765432]$", re.IGNORECASE)


def _canonical_call(raw):
    call = raw.strip().upper()
    if call in {"P", "PASS"}:
        return "PASS"
    if call in {"X", "DBL"}:
        return "X"
    if call in {"XX", "RDBL"}:
        return "XX"
    return call.replace("NT", "N")


def _canonical_card(raw):
    card = raw.strip().upper()
    if not CARD_RE.fullmatch(card):
        raise ValueError(f"Invalid bridge card token: {raw}")
    return card


def _parse_tags(block):
    tags = {}
    for line in block.splitlines():
        match = re.match(r'^\[(\w+)\s+"(.*)"\]\s*$', line.strip())
        if match:
            tags[match.group(1)] = match.group(2)
    return tags


def _parse_deal(deal):
    match = re.match(r"^([NESW]):(.+)$", deal.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid PBN Deal tag: {deal}")
    first_seat = match.group(1).upper()
    hands_text = match.group(2).split()
    if len(hands_text) != 4:
        raise ValueError("PBN Deal must contain four hands")
    first_index = SEATS.index(first_seat)
    hands = {}
    for offset, hand_text in enumerate(hands_text):
        seat = SEATS[(first_index + offset) % 4]
        cards = []
        suit_parts = hand_text.split(".")
        if len(suit_parts) != 4:
            raise ValueError(f"Invalid bridge hand: {hand_text}")
        for suit, ranks in zip(SUITS, suit_parts):
            for rank in ranks.upper():
                if rank == "-":
                    continue
                if rank not in RANKS:
                    raise ValueError(f"Invalid bridge rank: {rank}")
                cards.append(f"{suit}{rank}")
        if len(cards) != 13:
            raise ValueError(f"Bridge hand for {seat} has {len(cards)} cards")
        hands[seat] = cards
    all_cards = [card for cards in hands.values() for card in cards]
    if set(all_cards) != ALL_CARDS or len(all_cards) != len(set(all_cards)):
        raise ValueError("Bridge deal must contain each card exactly once")
    return hands


def _find_section_lines(block, section_name):
    lines = block.splitlines()
    found = False
    section_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(rf"^\[{section_name}\b", stripped, flags=re.IGNORECASE):
            found = True
            continue
        if found and stripped.startswith("[") and stripped.endswith("]"):
            break
        if found:
            section_lines.append(stripped)
    return section_lines


def _parse_auction(block, tags):
    auction_lines = _find_section_lines(block, "Auction")
    if not auction_lines:
        auction = tags.get("Auction", "")
        auction_lines = [auction] if auction else []
    calls = []
    for line in auction_lines:
        line = re.sub(r"\{[^}]*\}", " ", line)
        line = line.split(";")[0]
        for raw in line.split():
            if raw in {"*", "-", "AP"}:
                continue
            call = _canonical_call(raw)
            if CALL_RE.fullmatch(call):
                calls.append(call)
            else:
                raise ValueError(f"Invalid bridge call: {raw}")
    return calls


def _parse_play(block, tags):
    play_lines = _find_section_lines(block, "Play")
    if not play_lines:
        play = tags.get("Play", "")
        play_lines = [play] if play else []
    cards = []
    for line in play_lines:
        line = re.sub(r"\{[^}]*\}", " ", line)
        line = line.split(";")[0]
        for raw in line.split():
            if raw in {"*", "-", "--"}:
                continue
            cards.append(_canonical_card(raw))
    if len(cards) != len(set(cards)):
        raise ValueError("Bridge play contains duplicate cards")
    return cards


def _bridge_block_to_tokens(block, source_path):
    tags = _parse_tags(block)
    if "Deal" not in tags:
        return None
    hands = _parse_deal(tags["Deal"])
    calls = _parse_auction(block, tags)
    played_cards = _parse_play(block, tags)
    if len(calls) < 4 and not played_cards:
        return None

    tokens = ["<bos>", "<bridge>", "view_complete"]
    dealer = tags.get("Dealer")
    if dealer:
        tokens.append(f"dealer:{dealer.upper()}")
    vulnerable = tags.get("Vulnerable")
    if vulnerable:
        tokens.append(f"vul:{vulnerable.upper().replace(' ', '_')}")
    contract = tags.get("Contract")
    if contract:
        tokens.append(f"contract:{contract.upper().replace(' ', '_')}")
    declarer = tags.get("Declarer")
    if declarer:
        tokens.append(f"declarer:{declarer.upper()}")

    for seat in SEATS:
        tokens.append(f"hand:{seat}:{''.join(hands[seat])}")
    tokens.extend(f"bid:{call}" for call in calls)
    tokens.extend(f"play:{card}" for card in played_cards)
    tokens.append("<eos>")

    metadata = {
        "event": tags.get("Event", "Unknown"),
        "site": tags.get("Site", "Unknown"),
        "date": tags.get("Date", "????.??.??"),
        "board": tags.get("Board"),
        "dealer": dealer,
        "vulnerable": vulnerable,
        "contract": contract,
        "declarer": declarer,
        "result": tags.get("Result"),
        "seat_count": 4,
        "view_type": "complete",
        "viewer_seat": None,
        "move_count": len(calls) + len(played_cards),
        "source_path": str(Path(source_path).resolve()),
    }
    return tokens, metadata


def iter_pbn_files(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.name.lower().endswith(".pbn"):
            yield str(path)
        return
    for root, _, files in os.walk(path):
        for name in sorted(files):
            if name.lower().endswith(".pbn"):
                yield str(Path(root) / name)


def parse_pbn_to_tokens(pbn_path, max_games=None):
    if not os.path.exists(pbn_path):
        print(f"[Error] Bridge PBN file not found: {pbn_path}")
        return
    parsed = 0
    with open(pbn_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    blocks = re.split(r"\n\s*\n(?=\[)", content)
    for block in blocks:
        if not block.strip():
            continue
        try:
            parsed_entry = _bridge_block_to_tokens(block, pbn_path)
        except ValueError as exc:
            print(f"[Warning] Skipping invalid bridge board in {os.path.basename(pbn_path)}: {exc}")
            continue
        if parsed_entry is None:
            continue
        parsed += 1
        yield parsed_entry
        if max_games and parsed >= max_games:
            return


def parse_bridge_inputs(input_path, max_games=None):
    parsed = 0
    for pbn_path in iter_pbn_files(input_path):
        remaining = None if max_games is None else max_games - parsed
        if remaining is not None and remaining <= 0:
            break
        for tokens, metadata in parse_pbn_to_tokens(pbn_path, max_games=remaining):
            parsed += 1
            yield tokens, metadata
