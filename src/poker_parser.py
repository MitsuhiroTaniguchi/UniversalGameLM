import os
import random
import collections
import re
import ast
import hashlib
from pathlib import Path

# Card representation: Values 2-A, Suits h, d, c, s
VALUES = "23456789TJQKA"
SUITS = "hdcs"
FULL_DECK = [f"{v}{s}" for v in VALUES for s in SUITS]
CARD_RE = re.compile(r"[2-9TJQKA][hdcs]", re.IGNORECASE)

class PokerHandSimulator:
    """
    Simulates No-Limit Texas Hold'em hands with realistic players
    and generates unified action/state tokens.
    """
    def __init__(self, num_seats=6):
        self.num_seats = num_seats
        self.deck = [f"{v}{s}" for v in VALUES for s in SUITS]

    def _eval_hand(self, cards):
        """
        Simple, robust poker hand ranker.
        Returns a tuple: (rank_score, tie_breakers)
        """
        # Parse values and suits
        values_decoded = {"2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
        parsed = sorted([(values_decoded[c[0]], c[1]) for c in cards], reverse=True)
        
        vals = [p[0] for p in parsed]
        suits = [p[1] for p in parsed]
        
        # Flush check
        suit_counts = collections.Counter(suits)
        flush_suit = None
        for s, count in suit_counts.items():
            if count >= 5:
                flush_suit = s
                break
                
        # Straight check
        uniq_vals = sorted(list(set(vals)), reverse=True)
        straight_high = None
        # Handle Ace-low straight (A,2,3,4,5)
        if 14 in uniq_vals:
            uniq_vals_ace_low = uniq_vals + [1]
        else:
            uniq_vals_ace_low = uniq_vals
            
        for i in range(len(uniq_vals_ace_low) - 4):
            if uniq_vals_ace_low[i] - uniq_vals_ace_low[i+4] == 4:
                straight_high = uniq_vals_ace_low[i]
                break
                
        # Flush and Straight Flush
        if flush_suit:
            flush_cards = sorted([p[0] for p in parsed if p[1] == flush_suit], reverse=True)
            # Straight flush check inside flush cards
            flush_uniq = sorted(list(set(flush_cards)), reverse=True)
            if 14 in flush_uniq:
                flush_uniq_ace_low = flush_uniq + [1]
            else:
                flush_uniq_ace_low = flush_uniq
                
            for i in range(len(flush_uniq_ace_low) - 4):
                if flush_uniq_ace_low[i] - flush_uniq_ace_low[i+4] == 4:
                    return (8, flush_uniq_ace_low[i]) # Straight Flush
            return (5, flush_cards[:5]) # Flush
            
        if straight_high:
            return (4, straight_high) # Straight
            
        # Group duplicates
        counts = collections.Counter(vals)
        most_common = counts.most_common()
        
        # Four of a Kind
        if most_common[0][1] == 4:
            kicker = max([v for v in vals if v != most_common[0][0]])
            return (7, (most_common[0][0], kicker))
            
        # Full House
        if most_common[0][1] == 3 and len(most_common) > 1 and most_common[1][1] >= 2:
            return (6, (most_common[0][0], most_common[1][0]))
            
        # Three of a Kind
        if most_common[0][1] == 3:
            kickers = sorted([v for v in vals if v != most_common[0][0]], reverse=True)[:2]
            return (3, (most_common[0][0], kickers))
            
        # Two Pair
        if most_common[0][1] == 2 and len(most_common) > 1 and most_common[1][1] == 2:
            kicker = max([v for v in vals if v != most_common[0][0] and v != most_common[1][0]])
            return (2, (most_common[0][0], most_common[1][0], kicker))
            
        # One Pair
        if most_common[0][1] == 2:
            kickers = sorted([v for v in vals if v != most_common[0][0]], reverse=True)[:3]
            return (1, (most_common[0][0], kickers))
            
        # High Card
        return (0, vals[:5])

    def get_best_hand(self, hole_cards, community_cards):
        """Finds the best 5-card combination from 7 total cards."""
        all_cards = hole_cards + community_cards
        # Evaluate all 7 cards by selecting the best 5
        # For simplicity in this emulator, we evaluate the 7-card pool directly
        return self._eval_hand(all_cards)

    def _score_key(self, score):
        """Normalizes hand scores so same-class hands compare by kickers too."""
        rank, tie_breakers = score
        if isinstance(tie_breakers, int):
            tie_breakers = (tie_breakers,)
        elif isinstance(tie_breakers, list):
            tie_breakers = tuple(tie_breakers)
        else:
            flattened = []
            for value in tie_breakers:
                if isinstance(value, list):
                    flattened.extend(value)
                else:
                    flattened.append(value)
            tie_breakers = tuple(flattened)
        return (rank, *tie_breakers)

    def simulate_hand(self):
        """Simulates one No-Limit Hold'em hand and yields tokens."""
        # Shuffle deck
        deck = list(self.deck)
        random.shuffle(deck)
        
        # 1. Deal Hole Cards to active seats (we use seats 1 to 6)
        hands = {}
        for s in range(1, self.num_seats + 1):
            hands[s] = [deck.pop(), deck.pop()]
            
        # State tokens list
        tokens = ["<bos>", "<poker>"]
        
        # 2. Blinds Posting. Hole cards are intentionally not emitted here:
        # pre-showdown action tokens must not leak hidden information.
        sb_amt, bb_amt = 10, 20
        tokens.append(f"SB:{sb_amt}")
        tokens.append(f"BB:{bb_amt}")
        
        # Community Cards
        flop = [deck.pop(), deck.pop(), deck.pop()]
        turn = [deck.pop()]
        river = [deck.pop()]
        
        # Simulate Betting Actions across rounds
        active_players = list(range(1, self.num_seats + 1))
        
        # Pre-flop betting
        # Seats 1-6 profiles: 1 is SB, 2 is BB, 3 is UTG, 4 is HJ, 5 is CO, 6 is BTN
        # Simulated preflop actions
        for s in [3, 4, 5, 6, 1, 2]:
            if s not in active_players:
                continue
            card1_val = hands[s][0][0]
            card2_val = hands[s][1][0]
            
            # Simple bot logic: fold weak hands
            is_strong = any(v in "AKQJT" for v in [card1_val, card2_val])
            is_pair = (card1_val == card2_val)
            
            if is_strong or is_pair:
                action = "c"  # call
                if is_pair and card1_val in "AKQJ":
                    action = "r60"  # raise to 60
            else:
                action = "f"  # fold
                active_players.remove(s)
                
            tokens.append(f"P:{s}:{action}")
            
        # Flop betting round (if at least 2 active players left)
        if len(active_players) >= 2:
            tokens.append(f"FLOP:{flop[0]}{flop[1]}{flop[2]}")
            # Simulate check/bet
            for s in list(active_players):
                # Simple flop decisions
                has_pair = hands[s][0][0] in [c[0] for c in flop] or hands[s][1][0] in [c[0] for c in flop]
                if has_pair:
                    action = "b40"  # bet 40
                else:
                    action = "k"  # check
                tokens.append(f"F:{s}:{action}")
                
        # Turn betting round
        if len(active_players) >= 2:
            tokens.append(f"TURN:{turn[0]}")
            for s in list(active_players):
                tokens.append(f"T:{s}:k")
                
        # River betting round
        if len(active_players) >= 2:
            tokens.append(f"RIVER:{river[0]}")
            # Final river decisions: aggressive bet from BTN or SB
            for s in list(active_players):
                if s == active_players[-1]:
                    action = "b100"
                else:
                    action = "c"
                tokens.append(f"R:{s}:{action}")
                
        # Showdown & Determine Winner
        winner = None
        if len(active_players) == 1:
            winner = active_players[0]
        else:
            # Evaluate hands
            best_score = None
            for s in active_players:
                tokens.append(f"SHOW:{s}:{hands[s][0]}{hands[s][1]}")
                score = self._score_key(self.get_best_hand(hands[s], flop + turn + river))
                if best_score is None or score > best_score:
                    best_score = score
                    winner = s
                    
        tokens.append(f"WINNER:{winner}")
        tokens.append("<eos>")
        
        metadata = {
            "num_players": self.num_seats,
            "flop": "".join(flop),
            "turn": turn[0],
            "river": river[0],
            "winner": f"Player {winner}",
            "source": "synthetic_simulator",
            "move_count": len(tokens) - 3 # excluding bos/eos/game token
        }
        
        return tokens, metadata

def generate_poker_dataset(n_hands=100):
    """
    Generates a dataset of simulated No-Limit Texas Hold'em hands.
    """
    simulator = PokerHandSimulator()
    print(f"[Simulating Poker] Generating {n_hands} Hold'em hand histories...")
    for _ in range(n_hands):
        yield simulator.simulate_hand()
    print(f"[Success] Generated {n_hands} simulated Poker hands.")

def _sanitize_poker_action(action):
    action = re.sub(r"#.*$", "", action.strip())
    action = re.sub(r"\s+", "_", action.lower())
    return re.sub(r"[^a-z0-9_:\-.]+", "", action)

def _is_public_phh_action(action):
    normalized = action.strip().lower()
    compact = re.sub(r"\s+", " ", normalized)
    private_patterns = (
        r"^deal_hole\b",
        r"^hole\b",
        r"^dh\b",
        r"^d dh\b",
        r"^d\.dh\b",
        r"^show_or_muck_hole_cards\b",
    )
    return not any(re.match(pattern, compact) for pattern in private_patterns)


def _parse_phh_scalar(text, name):
    match = re.search(rf"^{re.escape(name)}\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return None
    value = match.group(1).split("#", 1)[0].strip().rstrip(",")
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip("\"'")


def _extract_actions_literals(text):
    match = re.search(r"^actions\s*=\s*\[", text, flags=re.MULTILINE)
    if not match:
        return []
    start = match.end() - 1
    depth = 0
    quote = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                literal = text[start:index + 1]
                try:
                    actions = ast.literal_eval(literal)
                except Exception:
                    return []
                return [str(a) for a in actions]
    return []


def _phh_state_tokens(text):
    tokens = []
    for field in ("variant", "ante_trimming_status", "betting_type"):
        value = _parse_phh_scalar(text, field)
        if value is not None:
            tokens.append(f"{field.upper()}:{str(value).lower().replace(' ', '_')}")
    for field in ("antes", "blinds_or_straddles", "min_bet", "starting_stacks"):
        value = _parse_phh_scalar(text, field)
        if value is not None:
            compact = re.sub(r"\s+", "", repr(value).lower())
            tokens.append(f"{field.upper()}:{compact}")
    return tokens


def _normalize_card(card):
    card = card.strip()
    if not re.fullmatch(r"[2-9TJQKA][hdcs]", card, flags=re.IGNORECASE):
        raise ValueError(f"Invalid poker card: {card}")
    return card[0].upper() + card[1].lower()


def _extract_cards(text):
    return [_normalize_card(card) for card in CARD_RE.findall(text)]


def _cards_token(cards):
    return "".join(cards)


def _seat_from_action(action):
    match = re.search(r"\bp(\d+)\b", action.lower())
    return int(match.group(1)) if match else None


def _private_hole_from_action(action):
    compact = re.sub(r"\s+", " ", action.strip().lower())
    if not (
        re.match(r"^d dh\b", compact)
        or re.match(r"^dh\b", compact)
        or re.match(r"^deal_hole\b", compact)
        or re.match(r"^hole\b", compact)
    ):
        return None
    seat = _seat_from_action(action)
    cards = _extract_cards(action)
    if seat is None or len(cards) < 2:
        return None
    return seat, cards[:2]


def _players_from_state_and_actions(state_tokens, actions, private_holes):
    seats = set(private_holes)
    for action in actions:
        seat = _seat_from_action(action)
        if seat is not None:
            seats.add(seat)
    return sorted(seats)


def _public_action_token(action):
    if not _is_supported_public_action(action):
        if _extract_cards(action):
            raise ValueError(f"Rejecting unknown card-bearing PHH action: {action}")
        return None
    sanitized = _sanitize_poker_action(action)
    if not sanitized:
        return None
    if re.match(r"^p\d+_sm_-?$", sanitized):
        return sanitized.replace("_-", "_hidden")
    return sanitized


def _is_supported_public_action(action):
    compact = re.sub(r"\s+", " ", action.strip().lower())
    return bool(re.match(
        r"^(?:"
        r"p\d+ (?:cc|cbr|br|f|fold|check|call|raise|bet|sm|show|muck)\b|"
        r"(?:call|check|fold|bet|raise) p\d+\b|"
        r"(?:p\d+_)?(?:post_blind|post_ante|ante|blind)\b|"
        r"(?:d db|db|deal_board|board)\b|"
        r"(?:flop|turn|river)\b"
        r")",
        compact,
    ))


def _observed_cards_from_public_actions(actions):
    observed = []
    for action in actions:
        compact = re.sub(r"\s+", " ", action.strip().lower())
        if re.match(r"^(?:d db|db|deal_board|board)\b", compact):
            observed.extend(_extract_cards(action))
        elif re.match(r"^p\d+ sm\b", compact) and "-" not in compact:
            observed.extend(_extract_cards(action))
    return observed


def _assert_unique_cards(cards):
    duplicates = [card for card, count in collections.Counter(cards).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate cards in PHH hand: {duplicates}")


def _completion_seed(actions, state_tokens):
    digest = hashlib.sha256("|".join(state_tokens + actions).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _complete_private_holes(actions, state_tokens, private_holes):
    public_cards = _observed_cards_from_public_actions(actions)
    known_private_cards = [card for cards in private_holes.values() for card in cards]
    _assert_unique_cards(public_cards + known_private_cards)
    players = _players_from_state_and_actions(state_tokens, actions, private_holes)
    used = set(public_cards + known_private_cards)
    available = [card for card in FULL_DECK if card not in used]
    rng = random.Random(_completion_seed(actions, state_tokens))
    rng.shuffle(available)

    completed = {seat: list(cards) for seat, cards in private_holes.items()}
    for seat in players:
        if seat not in completed:
            completed[seat] = [available.pop(), available.pop()]

    deck = public_cards + [card for seat in players for card in completed[seat]] + available
    _assert_unique_cards(deck)
    return completed, deck


def _bracket_delta_outside_strings(line):
    delta = 0
    quote = None
    escaped = False
    for char in line:
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "#":
            break
        elif char == "[":
            delta += 1
        elif char == "]":
            delta -= 1
    return delta


def poker_view_entries(actions, state_tokens):
    private_holes = {}
    public_actions = []
    private_excluded = 0
    for action in actions:
        private = _private_hole_from_action(action)
        if private is not None:
            seat, cards = private
            if seat in private_holes:
                raise ValueError(f"Duplicate private deal for seat p{seat}")
            private_holes[seat] = cards
            private_excluded += 1
            continue
        if not _is_public_phh_action(action):
            private_excluded += 1
            continue
        token = _public_action_token(action)
        if token:
            public_actions.append(token)

    if len(public_actions) < 2:
        return []

    completed_holes, deck = _complete_private_holes(actions, state_tokens, private_holes)
    player_count = len(completed_holes)
    if player_count < 2:
        return []
    view_rows_per_hand = player_count + 2
    base_metadata = {
        "seat_count": player_count,
        "player_count": player_count,
        "view_rows_per_hand": view_rows_per_hand,
        "move_count": len(public_actions),
        "private_actions_excluded": private_excluded,
        "completion_policy": "uniform_unknown_cards_v1",
        "completion_seed": _completion_seed(actions, state_tokens),
    }

    entries = [
        (
            ["<bos>", "<poker>", "view_complete"] + state_tokens + public_actions + ["<eos>"],
            {**base_metadata, "view_type": "complete", "viewer_seat": None},
        )
    ]

    for seat in sorted(completed_holes):
        hole_token = f"private_cards:p{seat}:{_cards_token(completed_holes[seat])}"
        entries.append(
            (
                ["<bos>", "<poker>", f"view_imperfect_p{seat}", hole_token] + state_tokens + public_actions + ["<eos>"],
                {**base_metadata, "view_type": "imperfect", "viewer_seat": seat},
            )
        )

    private_tokens = [
        f"private_cards:p{seat}:{_cards_token(cards)}"
        for seat, cards in sorted(completed_holes.items())
    ]
    entries.append(
        (
            ["<bos>", "<poker>", "view_omniscient"]
            + private_tokens
            + [f"deck:{_cards_token(deck)}"]
            + state_tokens
            + public_actions
            + ["<eos>"],
            {**base_metadata, "view_type": "omniscient", "viewer_seat": None},
        )
    )
    return entries

def iter_phh_action_lists(phh_path):
    """Streams PHH action arrays without loading the whole file."""
    state_lines = []
    action_lines = []
    in_actions = False
    bracket_depth = 0
    with open(phh_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not in_actions:
                if re.match(r"^actions\s*=", line.strip()):
                    in_actions = True
                    action_lines = [line]
                    bracket_depth = _bracket_delta_outside_strings(line)
                    if bracket_depth <= 0:
                        text = "".join(state_lines + action_lines)
                        actions = _extract_actions_literals(text)
                        if actions:
                            yield actions, _phh_state_tokens(text)
                        state_lines = []
                        action_lines = []
                        in_actions = False
                else:
                    state_lines.append(line)
                continue

            action_lines.append(line)
            bracket_depth += _bracket_delta_outside_strings(line)
            if bracket_depth <= 0:
                text = "".join(state_lines + action_lines)
                actions = _extract_actions_literals(text)
                if actions:
                    yield actions, _phh_state_tokens(text)
                state_lines = []
                action_lines = []
                in_actions = False

def iter_phh_files(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.name.lower().endswith((".phh", ".phhs")):
            yield str(path)
        return
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            if name.lower().endswith((".phh", ".phhs")):
                yield str(Path(root) / name)

def parse_phh_to_tokens(phh_path, max_hands=None):
    """
    Parses a simple PHH/PHHS file and yields public action tokens.

    PHH is TOML-like. For safety, private hole-card deal actions are excluded
    because they would leak hidden information before player decisions.
    """
    if not os.path.exists(phh_path):
        print(f"[Error] PHH file not found: {phh_path}")
        return

    parsed = 0
    for source_file in iter_phh_files(phh_path):
        for actions, state_tokens in iter_phh_action_lists(source_file):
            try:
                view_entries = poker_view_entries(actions, state_tokens)
            except ValueError as exc:
                print(f"[Warning] Skipping invalid PHH hand in {os.path.basename(source_file)}: {exc}")
                continue
            if not view_entries:
                continue

            parsed += 1
            for tokens, view_metadata in view_entries:
                metadata = {
                    **view_metadata,
                    "source": "phh",
                    "filename": os.path.basename(source_file),
                    "hand_index": parsed,
                    "view_group_id": f"{Path(source_file).resolve()}#{parsed}",
                    "source_path": str(Path(source_file).resolve()),
                }
                yield tokens, metadata

            if max_hands and parsed >= max_hands:
                return

if __name__ == "__main__":
    # Test simulator
    for i, (tokens, meta) in enumerate(generate_poker_dataset(n_hands=3)):
        print(f"\nHand #{i+1} Metadata: {meta}")
        print(f"Tokens (length {len(tokens)}): {tokens[:15]} ... {tokens[-5:]}")
