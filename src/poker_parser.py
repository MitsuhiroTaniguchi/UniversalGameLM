import os
import random
import collections
import re

# Card representation: Values 2-A, Suits h, d, c, s
VALUES = "23456789TJQKA"
SUITS = "hdcs"

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
    return re.sub(r"\s+", "_", action.strip().lower())

def _is_public_phh_action(action):
    normalized = action.strip().lower()
    private_prefixes = (
        "deal_hole",
        "dh ",
        "hole",
        "show_or_muck_hole_cards",
    )
    return not normalized.startswith(private_prefixes)

def parse_phh_to_tokens(phh_path, max_hands=None):
    """
    Parses a simple PHH/PHHS file and yields public action tokens.

    PHH is TOML-like. For safety, private hole-card deal actions are excluded
    because they would leak hidden information before player decisions.
    """
    if not os.path.exists(phh_path):
        print(f"[Error] PHH file not found: {phh_path}")
        return

    with open(phh_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Supports one PHH document or multiple documents separated by blank-table starts.
    blocks = re.split(r"\n(?=\s*\[\[?hand)", content)
    if len(blocks) == 1:
        blocks = [content]

    parsed = 0
    for block in blocks:
        actions = re.findall(r'"([^"]+)"', block)
        public_actions = [_sanitize_poker_action(a) for a in actions if _is_public_phh_action(a)]
        if len(public_actions) < 2:
            continue

        parsed += 1
        tokens = ["<bos>", "<poker>"] + public_actions + ["<eos>"]
        metadata = {
            "source": "phh",
            "filename": os.path.basename(phh_path),
            "move_count": len(public_actions),
            "private_actions_excluded": len(actions) - len(public_actions),
        }
        yield tokens, metadata

        if max_hands and parsed >= max_hands:
            break

if __name__ == "__main__":
    # Test simulator
    for i, (tokens, meta) in enumerate(generate_poker_dataset(n_hands=3)):
        print(f"\nHand #{i+1} Metadata: {meta}")
        print(f"Tokens (length {len(tokens)}): {tokens[:15]} ... {tokens[-5:]}")
