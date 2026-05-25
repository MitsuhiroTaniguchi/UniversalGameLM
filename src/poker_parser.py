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
POKER_MOVE_ACTIONS = {
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


def poker_action_count(tokens):
    return sum(1 for token in tokens if token in POKER_MOVE_ACTIONS)

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
        most_common = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
        
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

    def _postflop_betting_round(self, active_players, should_bet, should_continue, should_raise, bet_amount):
        order = list(active_players)
        bettor = next((seat for seat in order if should_bet(seat)), None)
        if bettor is None:
            return [token for seat in order for token in (f"seat:p{seat}", "act:check")], order

        tokens = []
        remaining = list(order)
        bettor_index = order.index(bettor)
        for seat in order[:bettor_index]:
            tokens.extend([f"seat:p{seat}", "act:check"])
        tokens.extend([f"seat:p{bettor}", "act:bet"] + _number_digit_tokens("AMT", bet_amount))
        current_amount = bet_amount
        raised_by = None
        action_order = order[bettor_index + 1:] + order[:bettor_index]
        acted_before_raise = []
        for seat in action_order:
            if raised_by is None and should_raise(seat):
                current_amount = bet_amount * 2
                raised_by = seat
                tokens.extend([f"seat:p{seat}", "act:raise"] + _number_digit_tokens("AMT", current_amount))
            elif should_continue(seat):
                tokens.extend([f"seat:p{seat}", "act:call"] + _number_digit_tokens("AMT", current_amount))
                if raised_by is None:
                    acted_before_raise.append(seat)
            else:
                tokens.extend([f"seat:p{seat}", "act:fold"])
                remaining.remove(seat)

        if raised_by is not None:
            for seat in [bettor] + acted_before_raise:
                if seat not in remaining:
                    continue
                if should_continue(seat):
                    tokens.extend([f"seat:p{seat}", "act:call"] + _number_digit_tokens("AMT", current_amount))
                else:
                    tokens.extend([f"seat:p{seat}", "act:fold"])
                    remaining.remove(seat)
        return tokens, remaining

    def simulate_hand(self, return_state=False):
        """Simulates one No-Limit Hold'em hand and yields tokens."""
        # Shuffle deck
        deck = list(self.deck)
        random.shuffle(deck)
        
        # 1. Deal Hole Cards to active seats (we use seats 1 to 6)
        hands = {}
        for s in range(1, self.num_seats + 1):
            hands[s] = [deck.pop(), deck.pop()]
            
        # State tokens list
        public_tokens = []
        
        # 2. Blinds Posting. Hole cards are intentionally not emitted here:
        # pre-showdown action tokens must not leak hidden information.
        sb_amt, bb_amt = 10, 20
        public_tokens.extend(["act:preflop", "seat:p1", "act:post_small_blind"] + _number_digit_tokens("AMT", sb_amt))
        public_tokens.extend(["seat:p2", "act:post_big_blind"] + _number_digit_tokens("AMT", bb_amt))
        
        # Community cards are dealt only when the street is reached. If everyone
        # folds preflop, those cards remain in the undealt deck for omniscient views.
        flop = []
        turn = []
        river = []
        
        # Simulate Betting Actions across rounds
        active_players = list(range(1, self.num_seats + 1))
        
        # Pre-flop betting
        # Seats 1-6 profiles: 1 is SB, 2 is BB, 3 is UTG, 4 is HJ, 5 is CO, 6 is BTN
        preflop_order = [seat for seat in list(range(3, self.num_seats + 1)) + [1, 2] if seat in active_players]
        preflop_raised = False
        preflop_raise_amount = bb_amt
        preflop_raiser = None
        acted_before_preflop_raise = []
        for s in preflop_order:
            if s not in active_players:
                continue
            card1_val = hands[s][0][0]
            card2_val = hands[s][1][0]
            
            # Simple bot logic: fold weak hands
            is_strong = any(v in "AKQJT" for v in [card1_val, card2_val])
            is_pair = (card1_val == card2_val)
            
            if s == 2:
                if preflop_raised:
                    if is_pair and card1_val in "AKQ":
                        preflop_raise_amount *= 2
                        preflop_raiser = s
                        action_tokens = ["act:raise"] + _number_digit_tokens("AMT", preflop_raise_amount)
                    elif is_strong or is_pair:
                        action_tokens = ["act:call"] + _number_digit_tokens("AMT", preflop_raise_amount)
                    else:
                        action_tokens = ["act:fold"]
                        active_players.remove(s)
                else:
                    if is_pair and card1_val in "AKQ":
                        preflop_raise_amount = 60
                        preflop_raiser = s
                        action_tokens = ["act:raise"] + _number_digit_tokens("AMT", preflop_raise_amount)
                        preflop_raised = True
                    else:
                        action_tokens = ["act:check"]
            elif is_pair and card1_val in "AKQJ":
                preflop_raise_amount = 60
                action_tokens = ["act:raise"] + _number_digit_tokens("AMT", preflop_raise_amount)
                preflop_raised = True
                preflop_raiser = s
            elif is_strong or is_pair:
                amount = preflop_raise_amount if preflop_raised else bb_amt
                action_tokens = ["act:call"] + _number_digit_tokens("AMT", amount)
                if not preflop_raised:
                    acted_before_preflop_raise.append(s)
            else:
                action_tokens = ["act:fold"]
                active_players.remove(s)
                
            public_tokens.extend([f"seat:p{s}"] + action_tokens)

        if preflop_raised:
            for s in acted_before_preflop_raise:
                if s not in active_players or s == preflop_raiser:
                    continue
                card1_val = hands[s][0][0]
                card2_val = hands[s][1][0]
                is_strong = any(v in "AKQJT" for v in [card1_val, card2_val])
                is_pair = card1_val == card2_val
                if is_strong or is_pair:
                    public_tokens.extend([f"seat:p{s}", "act:call"] + _number_digit_tokens("AMT", preflop_raise_amount))
                else:
                    public_tokens.extend([f"seat:p{s}", "act:fold"])
                    active_players.remove(s)
            
        # Flop betting round (if at least 2 active players left)
        if len(active_players) >= 2:
            flop = [deck.pop(), deck.pop(), deck.pop()]
            public_tokens.extend(["act:flop"] + [f"card:{card}" for card in flop])
            flop_ranks = {card[0] for card in flop}
            round_tokens, active_players = self._postflop_betting_round(
                active_players,
                lambda seat: hands[seat][0][0] in flop_ranks or hands[seat][1][0] in flop_ranks,
                lambda seat: any(card[0] in "AKQJT" for card in hands[seat]) or any(card[0] in flop_ranks for card in hands[seat]),
                lambda seat: hands[seat][0][0] == hands[seat][1][0] and hands[seat][0][0] in "AKQJT",
                40,
            )
            public_tokens.extend(round_tokens)
                
        # Turn betting round
        if len(active_players) >= 2:
            turn = [deck.pop()]
            public_tokens.extend(["act:turn", f"card:{turn[0]}"])
            turn_rank = turn[0][0]
            turn_bettor = next(
                (seat for seat in active_players if any(card[0] == turn_rank for card in hands[seat])),
                None,
            )
            round_tokens, active_players = self._postflop_betting_round(
                active_players,
                lambda seat: seat == turn_bettor,
                lambda seat: any(card[0] in "AKQJT" for card in hands[seat]) or any(card[0] == turn_rank for card in hands[seat]),
                lambda seat: hands[seat][0][0] == hands[seat][1][0] and hands[seat][0][0] in "AKQ",
                80,
            )
            public_tokens.extend(round_tokens)
                
        # River betting round
        if len(active_players) >= 2:
            river = [deck.pop()]
            public_tokens.extend(["act:river", f"card:{river[0]}"])
            river_rank = river[0][0]
            river_bettor = next(
                (seat for seat in active_players if any(card[0] == river_rank for card in hands[seat])),
                active_players[0],
            )
            round_tokens, active_players = self._postflop_betting_round(
                active_players,
                lambda seat: seat == river_bettor,
                lambda seat: any(card[0] in "AKQJT" for card in hands[seat]) or any(card[0] == river_rank for card in hands[seat]),
                lambda seat: hands[seat][0][0] == hands[seat][1][0] and hands[seat][0][0] in "AKQ",
                100,
            )
            public_tokens.extend(round_tokens)
                
        # Showdown & Determine Winner
        winner = None
        if len(active_players) == 1:
            winner = active_players[0]
        else:
            # Evaluate hands
            best_score = None
            for s in active_players:
                public_tokens.extend([f"showdown:p{s}", f"card:{hands[s][0]}", f"card:{hands[s][1]}"])
                score = self._score_key(self.get_best_hand(hands[s], flop + turn + river))
                if best_score is None or score > best_score:
                    best_score = score
                    winner = s
                    
        public_tokens.append(f"winner:p{winner}")
        tokens = ["<bos>", "<poker>", "view_complete"] + public_tokens + ["<eos>"]
        
        metadata = {
            "num_players": self.num_seats,
            "flop": "".join(flop) or None,
            "turn": turn[0] if turn else None,
            "river": river[0] if river else None,
            "winner": f"Player {winner}",
            "source": "synthetic_simulator",
            "seat_count": self.num_seats,
            "view_type": "complete",
            "viewer_seat": None,
            "move_count": poker_action_count(public_tokens),
        }
        if return_state:
            return tokens, metadata, hands, deck
        return tokens, metadata

    def simulate_hand_views(self):
        complete_tokens, base_metadata, hands, undealt_cards = self.simulate_hand(return_state=True)
        view_group_id = f"synthetic:{hashlib.sha256(' '.join(complete_tokens).encode('utf-8')).hexdigest()[:16]}"
        base_metadata = {
            **base_metadata,
            "view_group_id": view_group_id,
            "view_rows_per_hand": self.num_seats + 2,
        }
        yield complete_tokens, {**base_metadata, "view_type": "complete", "viewer_seat": None}
        public_body = complete_tokens[3:-1]
        for seat in range(1, self.num_seats + 1):
            yield (
                ["<bos>", "<poker>", f"view_imperfect_p{seat}"]
                + _private_card_tokens(seat, hands[seat])
                + public_body
                + ["<eos>"],
                {**base_metadata, "view_type": "imperfect", "viewer_seat": seat},
            )
        private_tokens = []
        for seat in range(1, self.num_seats + 1):
            private_tokens.extend(_private_card_tokens(seat, hands[seat]))
        yield (
            ["<bos>", "<poker>", "view_omniscient"]
            + private_tokens
            + _undealt_card_tokens(undealt_cards)
            + public_body
            + ["<eos>"],
            {**base_metadata, "view_type": "omniscient", "viewer_seat": None},
        )

def generate_poker_dataset(n_hands=100):
    """
    Generates a dataset of simulated No-Limit Texas Hold'em hands.
    """
    simulator = PokerHandSimulator()
    print(f"[Simulating Poker] Generating {n_hands} Hold'em hand histories...")
    for _ in range(n_hands):
        yield from simulator.simulate_hand_views()
    print(f"[Success] Generated {n_hands} simulated Poker hands.")

def _sanitize_poker_action(action):
    action = re.sub(r"#.*$", "", action.strip())
    action = re.sub(r"\s+", "_", action.lower())
    return re.sub(r"[^a-z0-9_:\-.]+", "", action)


def _number_digit_tokens(prefix, value):
    text = str(value)
    tokens = []
    for char in text:
        if char.isdigit():
            tokens.append(f"{prefix}:{char}")
        elif char == ".":
            tokens.append(f"{prefix}:dot")
        elif char == "-":
            tokens.append(f"{prefix}:neg")
    return tokens or [f"{prefix}:0"]


def _structured_value_tokens(field, value):
    tokens = [f"{field.upper()}:BEGIN"]
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            if index:
                tokens.append(f"{field.upper()}:SEP")
            tokens.extend(_number_digit_tokens("NUM", item))
    else:
        tokens.extend(_number_digit_tokens("NUM", value))
    tokens.append(f"{field.upper()}:END")
    return tokens

def _is_public_phh_action(action):
    normalized = action.strip().lower()
    compact = re.sub(r"\s+", " ", normalized)
    private_patterns = (
        r"^deal_hole\b",
        r"^hole\b",
        r"^dh\b",
        r"^d dh\b",
        r"^d\.dh\b",
    )
    return not any(re.match(pattern, compact) for pattern in private_patterns)


def _parse_phh_scalar(text, name):
    matches = list(re.finditer(rf"^{re.escape(name)}\s*=\s*(.+)$", text, flags=re.MULTILINE))
    if not matches:
        return None
    match = matches[-1]
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
            tokens.extend(_structured_value_tokens(field, value))
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


def _private_card_tokens(seat, cards):
    return [f"private_card:p{seat}:{card}" for card in cards]


def _undealt_card_tokens(cards):
    return [f"undealt_card:{card}" for card in cards]


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


def _shown_hole_from_action(action):
    compact = re.sub(r"\s+", " ", action.strip().lower())
    if not re.match(r"^(?:p\d+ sm|show_or_muck_hole_cards p\d+)\b", compact) or "-" in compact:
        return None
    seat = _seat_from_action(action)
    cards = _extract_cards(action)
    if seat is None or len(cards) < 2:
        return None
    return seat, cards[:2]


def _players_from_actions(actions, private_holes):
    seats = set(private_holes)
    for action in actions:
        seat = _seat_from_action(action)
        if seat is not None:
            seats.add(seat)
    return sorted(seats)


def _public_action_tokens(action):
    if not _is_supported_public_action(action):
        if _extract_cards(action):
            raise ValueError(f"Rejecting unknown card-bearing PHH action: {action}")
        return []
    sanitized = _sanitize_poker_action(action)
    if not sanitized:
        return []
    if re.match(r"^p\d+_sm_-?$", sanitized):
        sanitized = sanitized.replace("_-", "_hidden")
    parts = [part for part in sanitized.split("_") if part]
    aliases = {
        "d": "deal",
        "db": "board",
        "cc": "call",
        "cbr": "raise",
        "br": "raise",
        "f": "fold",
        "sm": "show",
        "calls": "call",
        "checks": "check",
        "folds": "fold",
        "bets": "bet",
        "raises": "raise",
        "posts": "post",
        "shows": "show",
        "mucks": "muck",
    }
    parts = [aliases.get(part, part) for part in parts]

    seat = next((part for part in parts if re.fullmatch(r"p\d+", part)), None)
    amount_tokens = []
    card_tokens = []
    for part in parts:
        if re.fullmatch(r"\d+(?:\.\d+)?", part):
            amount_tokens.extend(_number_digit_tokens("AMT", part))
        elif CARD_RE.fullmatch(part):
            card_tokens.append(f"card:{_normalize_card(part)}")
        elif CARD_RE.findall(part) and "".join(CARD_RE.findall(part)).lower() == part.lower():
            card_tokens.extend(f"card:{card}" for card in _extract_cards(part))

    action_token = None
    if "deal" in parts and "board" in parts:
        action_token = "act:deal_board"
    elif {"show", "or", "muck", "hole", "cards"}.issubset(parts):
        action_token = "act:show"
    elif "post_blind" in parts or ("post" in parts and "blind" in parts and "small" not in parts and "big" not in parts):
        action_token = "act:post_blind"
    elif "post_ante" in parts:
        action_token = "act:post_ante"
    elif "post" in parts and "small" in parts and "blind" in parts:
        action_token = "act:post_small_blind"
    elif "post" in parts and "big" in parts and "blind" in parts:
        action_token = "act:post_big_blind"
    elif "post" in parts and "ante" in parts:
        action_token = "act:post_ante"
    elif "flop" in parts:
        action_token = "act:flop"
    elif "turn" in parts:
        action_token = "act:turn"
    elif "river" in parts:
        action_token = "act:river"
    else:
        for part in parts:
            if part in {"call", "check", "fold", "raise", "bet", "show", "muck", "blind", "ante", "board"}:
                action_token = f"act:{part}"
                break

    if action_token:
        tokens = []
        if seat:
            tokens.append(f"seat:{seat}")
        tokens.append(action_token)
        if "hidden" in parts:
            tokens.append("act:hidden")
        tokens.extend(amount_tokens)
        tokens.extend(card_tokens)
        return tokens

    tokens = []
    index = 0
    while index < len(parts):
        part = parts[index]
        if re.fullmatch(r"p\d+", part):
            tokens.append(f"seat:{part}")
        elif re.fullmatch(r"\d+(?:\.\d+)?", part):
            tokens.extend(_number_digit_tokens("AMT", part))
        elif CARD_RE.fullmatch(part):
            tokens.append(f"card:{_normalize_card(part)}")
        elif part == "post" and index + 2 < len(parts) and parts[index + 1] in {"small", "big"} and parts[index + 2] == "blind":
            tokens.append(f"act:post_{parts[index + 1]}_blind")
            index += 2
        elif part in {"post", "blind", "ante"} and index + 1 < len(parts):
            tokens.append(f"act:{part}_{parts[index + 1]}")
            index += 1
        elif part in {"deal", "board"} and index + 1 < len(parts):
            tokens.append(f"act:{part}_{parts[index + 1]}")
            index += 1
        else:
            tokens.append(f"act:{part}")
        index += 1
    return tokens


def _is_supported_public_action(action):
    compact = re.sub(r"\s+", " ", action.strip().lower())
    return bool(re.match(
        r"^(?:"
        r"p\d+ (?:cc|cbr|br|f|folds?|checks?|calls?|raises?|bets?|posts?|sm|shows?|mucks?)\b|"
        r"(?:calls?|checks?|folds?|bets?|raises?) p\d+\b|"
        r"p\d+ posts? (?:small blind|big blind|ante)\b|"
        r"show_or_muck_hole_cards p\d+\b|"
        r"(?:p\d+_)?(?:post_blind|post_ante|ante|blind|posts?)\b|"
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
    return observed


def _assert_unique_cards(cards):
    duplicates = [card for card, count in collections.Counter(cards).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate cards in PHH hand: {duplicates}")


def _completion_seed(actions, state_tokens):
    digest = hashlib.sha256("|".join(state_tokens + actions).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _observed_private_holes(actions, state_tokens, private_holes):
    public_cards = _observed_cards_from_public_actions(actions)
    known_private_cards = [card for cards in private_holes.values() for card in cards]
    _assert_unique_cards(public_cards + known_private_cards)
    players = _players_from_actions(actions, private_holes)
    missing = [seat for seat in players if seat not in private_holes]
    undealt_cards = []
    if not missing:
        undealt_cards = [
            card for card in FULL_DECK
            if card not in set(public_cards + known_private_cards)
        ]
    return {seat: list(cards) for seat, cards in private_holes.items()}, undealt_cards, players, missing


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
        shown = _shown_hole_from_action(action)
        if shown is not None:
            seat, cards = shown
            if seat in private_holes and private_holes[seat] != cards:
                raise ValueError(f"Shown cards contradict private deal for seat p{seat}")
            if seat not in private_holes:
                private_holes[seat] = cards
        if not _is_public_phh_action(action):
            private_excluded += 1
            continue
        public_actions.extend(_public_action_tokens(action))

    if len(public_actions) < 2:
        return []

    observed_holes, undealt_cards, players, missing_private_seats = _observed_private_holes(actions, state_tokens, private_holes)
    player_count = len(players)
    if player_count < 2:
        return []
    has_omniscient = not missing_private_seats and len(observed_holes) == player_count
    view_rows_per_hand = 1 + len(observed_holes) + (1 if has_omniscient else 0)
    base_metadata = {
        "seat_count": player_count,
        "player_count": player_count,
        "view_rows_per_hand": view_rows_per_hand,
        "move_count": poker_action_count(public_actions),
        "private_actions_excluded": private_excluded,
        "completion_policy": "observed_private_only_v1",
        "missing_private_seats": missing_private_seats,
        "completion_seed": _completion_seed(actions, state_tokens),
    }

    entries = [
        (
            ["<bos>", "<poker>", "view_complete"] + state_tokens + public_actions + ["<eos>"],
            {**base_metadata, "view_type": "complete", "viewer_seat": None},
        )
    ]

    for seat in sorted(observed_holes):
        entries.append(
            (
                ["<bos>", "<poker>", f"view_imperfect_p{seat}"]
                + _private_card_tokens(seat, observed_holes[seat])
                + state_tokens
                + public_actions
                + ["<eos>"],
                {**base_metadata, "view_type": "imperfect", "viewer_seat": seat},
            )
        )

    if has_omniscient:
        private_tokens = []
        for seat, cards in sorted(observed_holes.items()):
            private_tokens.extend(_private_card_tokens(seat, cards))
        entries.append(
            (
                ["<bos>", "<poker>", "view_omniscient"]
                + private_tokens
                + _undealt_card_tokens(undealt_cards)
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
