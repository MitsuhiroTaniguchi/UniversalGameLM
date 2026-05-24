import os
import re
from pathlib import Path


SEATS = ("N", "E", "S", "W")
SUITS = ("S", "H", "D", "C")
RANKS = "AKQJT98765432"
ALL_CARDS = {f"{r}{s.lower()}" for s in SUITS for r in RANKS}
CALL_RE = re.compile(r"^(?:PASS|P|X|XX|DBL|RDBL|[1-7](?:C|D|H|S|N|NT))$", re.IGNORECASE)
CARD_RE = re.compile(r"^(?:[SHDC][AKQJT98765432]|[AKQJT98765432][shdc])$", re.IGNORECASE)
STRAIN_ORDER = {"C": 0, "D": 1, "H": 2, "S": 3, "N": 4}


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
    if card[0] in "SHDC":
        return card[1] + card[0].lower()
    return card[0] + card[1].lower()


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
                cards.append(f"{rank}{suit.lower()}")
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


def _section_starter(tags, name, fallback=None):
    value = tags.get(name)
    if value and value.upper() in SEATS:
        return value.upper()
    return fallback


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


def _bid_rank(call):
    return int(call[0]), STRAIN_ORDER[call[1]]


def _validate_auction(calls, dealer):
    if dealer not in SEATS:
        raise ValueError("Bridge auction requires a valid dealer")
    highest_bid = None
    highest_bid_side = None
    contract_status = None
    consecutive_passes = 0
    ended = False
    for index, call in enumerate(calls):
        if ended:
            raise ValueError("Call after auction termination")
        seat = SEATS[(SEATS.index(dealer) + index) % 4]
        side = seat in ("N", "S")
        if call == "PASS":
            consecutive_passes += 1
            if highest_bid is None and consecutive_passes == 4:
                ended = True
            elif highest_bid is not None and consecutive_passes == 3:
                ended = True
            continue
        consecutive_passes = 0
        if call in {"X", "XX"}:
            if highest_bid is None:
                raise ValueError("Double/redouble before any bid")
            if call == "X":
                if highest_bid_side == side or contract_status is not None:
                    raise ValueError("Illegal double")
                contract_status = "X"
            elif contract_status != "X" or highest_bid_side != side:
                raise ValueError("Illegal redouble")
            else:
                contract_status = "XX"
            continue
        if highest_bid is not None and _bid_rank(call) <= _bid_rank(highest_bid):
            raise ValueError("Bridge bids must increase")
        highest_bid = call
        highest_bid_side = side
        contract_status = None
    if calls and not ended:
        raise ValueError("Auction is not terminated")
    return True


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


def _validate_play(played_cards, hands, leader):
    if not played_cards:
        return True
    if leader not in SEATS:
        raise ValueError("Bridge play requires a valid opening leader")
    if len(played_cards) % 4 != 0:
        raise ValueError("Bridge play must contain complete tricks")
    remaining = {seat: set(cards) for seat, cards in hands.items()}
    current_leader = leader
    for trick_start in range(0, len(played_cards), 4):
        trick_cards = played_cards[trick_start:trick_start + 4]
        led_suit = trick_cards[0][1]
        trick = []
        for offset, card in enumerate(trick_cards):
            seat = SEATS[(SEATS.index(current_leader) + offset) % 4]
            if card not in remaining[seat]:
                raise ValueError(f"{seat} cannot play {card}")
            if card[1] != led_suit and any(c[1] == led_suit for c in remaining[seat]):
                raise ValueError(f"{seat} revoked on {card}")
            remaining[seat].remove(card)
            trick.append((seat, card))
        current_leader = min(
            (item for item in trick if item[1][1] == led_suit),
            key=lambda item: RANKS.index(item[1][0]),
        )[0]
    return True


def _annotated_play(played_cards, hands, leader):
    _validate_play(played_cards, hands, leader)
    annotated = []
    if not played_cards:
        return annotated
    current_leader = leader
    for trick_start in range(0, len(played_cards), 4):
        trick_cards = played_cards[trick_start:trick_start + 4]
        led_suit = trick_cards[0][1]
        trick = []
        for offset, card in enumerate(trick_cards):
            seat = SEATS[(SEATS.index(current_leader) + offset) % 4]
            annotated.append((seat, card))
            trick.append((seat, card))
        current_leader = min(
            (item for item in trick if item[1][1] == led_suit),
            key=lambda item: RANKS.index(item[1][0]),
        )[0]
    return annotated


def _bridge_block_to_tokens(block, source_path):
    tags = _parse_tags(block)
    if "Deal" not in tags:
        return None
    hands = _parse_deal(tags["Deal"])
    dealer = tags.get("Dealer", "").upper()
    auction_starter = _section_starter(tags, "Auction", dealer)
    calls = _parse_auction(block, tags)
    _validate_auction(calls, auction_starter)
    played_cards = _parse_play(block, tags)
    play_starter = _section_starter(tags, "Play", None)
    played_by_seat = _annotated_play(played_cards, hands, play_starter)
    if len(calls) < 4 and not played_cards:
        return None

    context_tokens = []
    dealer = tags.get("Dealer")
    if dealer:
        context_tokens.append(f"dealer:{dealer.upper()}")
    vulnerable = tags.get("Vulnerable")
    if vulnerable:
        context_tokens.append(f"vul:{vulnerable.upper().replace(' ', '_')}")
    contract = tags.get("Contract")
    if contract:
        context_tokens.append(f"contract:{contract.upper().replace(' ', '_')}")
    declarer = tags.get("Declarer")
    if declarer:
        context_tokens.append(f"declarer:{declarer.upper()}")
    if play_starter:
        context_tokens.append(f"play_leader:{play_starter}")
    context_tokens.extend(f"bid:{call}" for call in calls)
    context_tokens.extend(f"play:{seat}:{card}" for seat, card in played_by_seat)

    base_metadata = {
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
        "bridge_auction_validated": True,
        "bridge_play_validated": True,
    }
    entries = [
        (["<bos>", "<bridge>", "view_complete"] + context_tokens + ["<eos>"], base_metadata)
    ]
    for seat in SEATS:
        entries.append((
            ["<bos>", "<bridge>", f"view_imperfect_{seat}", f"hand:{seat}:{''.join(hands[seat])}"] + context_tokens + ["<eos>"],
            {**base_metadata, "view_type": "imperfect", "viewer_seat": seat},
        ))
    omni_hands = [f"hand:{seat}:{''.join(hands[seat])}" for seat in SEATS]
    entries.append((
        ["<bos>", "<bridge>", "view_omniscient"] + omni_hands + context_tokens + ["<eos>"],
        {**base_metadata, "view_type": "omniscient", "viewer_seat": None},
    ))
    return entries


def iter_pbn_files(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.name.lower().endswith(".pbn"):
            yield str(path)
        return
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            if name.lower().endswith(".pbn"):
                yield str(Path(root) / name)


def parse_pbn_to_tokens(pbn_path, max_games=None):
    if not os.path.exists(pbn_path):
        print(f"[Error] Bridge PBN file not found: {pbn_path}")
        return
    parsed = 0
    current = []

    def emit_block(block):
        if not block.strip():
            return None
        try:
            return _bridge_block_to_tokens(block, pbn_path)
        except ValueError as exc:
            print(f"[Warning] Skipping invalid bridge board in {os.path.basename(pbn_path)}: {exc}")
            return None

    with open(pbn_path, "r", encoding="utf-8", errors="strict") as f:
        for line in f:
            if line.startswith("[Event ") and current:
                entries = emit_block("".join(current))
                current = []
                if entries is not None:
                    parsed += 1
                    for tokens, metadata in entries:
                        metadata = {**metadata, "hand_index": parsed, "view_group_id": f"{Path(pbn_path).resolve()}#{parsed}"}
                        yield tokens, metadata
                    if max_games and parsed >= max_games:
                        return
            current.append(line)
    if current:
        entries = emit_block("".join(current))
        if entries is not None:
            parsed += 1
            for tokens, metadata in entries:
                metadata = {**metadata, "hand_index": parsed, "view_group_id": f"{Path(pbn_path).resolve()}#{parsed}"}
                yield tokens, metadata


def parse_bridge_inputs(input_path, max_games=None):
    parsed = 0
    for pbn_path in iter_pbn_files(input_path):
        remaining = None if max_games is None else max_games - parsed
        if remaining is not None and remaining <= 0:
            break
        for tokens, metadata in parse_pbn_to_tokens(pbn_path, max_games=remaining):
            parsed += 1
            yield tokens, metadata
