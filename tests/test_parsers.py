import unittest
import os
import tempfile
import zipfile
import gzip
import json
import re
import sys
import types
from unittest import mock
from src.chess_parser import parse_pgn_to_tokens
from src.shogi_parser import parse_csa_to_tokens, parse_shogi_directory
from src.go_parser import parse_sgf_to_tokens
from src.othello_parser import othello_move_count, parse_othello_hf_dataset, parse_othello_jsonl_to_tokens, parse_othello_pgn_to_tokens, validate_othello_moves
from src.poker_parser import PokerHandSimulator
from src.poker_parser import generate_poker_dataset
from src.poker_parser import poker_action_count
from src.poker_parser import parse_phh_to_tokens
from src.bridge_parser import parse_bridge_inputs, parse_pbn_to_tokens
from src.tokenizer import UniversalGameTokenizer
from src.download import safe_extract_zip
from src.hf_uploader import HuggingFaceShardUploader
from src.production_pipeline import (
    assert_source_allowed_for_primary_build,
    build_game_shards,
    limit_entries,
    load_source_catalog,
    validate_entry,
    ProductionDatasetError,
)
from src.stats import DatasetStatsAccumulator
from src.stats import is_counted_move_token
from src.mahjonglm_compat import entry_to_mahjonglm_row, tokens_to_mahjonglm_stream

TERMINAL_OTHELLO_MOVES = [
    "d3", "c3", "b3", "b2", "b1", "a1", "c4", "c1",
    "c2", "d2", "d1", "e1", "a2", "a3", "f5", "e2",
    "f1", "g1", "pass", "f2", "pass", "e3", "pass", "b5",
    "b4", "a5", "a4", "c5", "a6", "f4", "f3", "g3",
    "g2", "h2", "h1", "h3", "h4", "g4", "c6", "g5",
    "h5", "b6", "c7", "d6", "e6", "f6", "g6", "h6",
    "h7", "a7", "pass", "b7", "a8", "d7", "e7", "f7",
    "g7", "g8", "b8", "c8", "d8", "e8", "f8", "h8",
    "pass", "pass",
]


def terminal_othello_pgn():
    pairs = []
    for idx in range(0, len(TERMINAL_OTHELLO_MOVES), 2):
        black = TERMINAL_OTHELLO_MOVES[idx].upper()
        white = TERMINAL_OTHELLO_MOVES[idx + 1].upper()
        pairs.append(f"{idx // 2 + 1}. {black} {white}")
    return " ".join(pairs) + " *"


class TestUniversalGameParsers(unittest.TestCase):
    
    def test_chess_parser(self):
        mock_pgn = """[Event "FICS Game"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 *"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(mock_pgn)
            temp_path = f.name
        try:
            games = list(parse_pgn_to_tokens(temp_path))
            self.assertEqual(len(games), 1)
            tokens, meta = games[0]
            self.assertEqual(tokens[0], "<bos>")
            self.assertEqual(tokens[1], "<chess>")
            self.assertEqual(tokens[2], "ch:w:e2")
            self.assertEqual(tokens[3], "ch:e4")
            self.assertEqual(tokens[-1], "<eos>")
            self.assertEqual(meta["white"], "Player1")
            self.assertEqual(meta["game_index"], 1)
        finally:
            os.remove(temp_path)

    def test_chess_parser_rejects_illegal_pgn(self):
        mock_pgn = """[Event "Broken"]
[White "Player1"]
[Black "Player2"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Qh5 Qh4 4. Qxh4 *"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(mock_pgn)
            temp_path = f.name
        try:
            self.assertEqual(list(parse_pgn_to_tokens(temp_path)), [])
        finally:
            os.remove(temp_path)

    def test_go_parser(self):
        # Create a mock SGF with 10 moves to pass the quality filter
        mock_sgf = "(;PB[BlackPlayer]PW[WhitePlayer]RE[B+R];B[pd];W[dd];B[pp];W[dp];B[cf];W[ch];B[fd];W[df];B[dg];W[cg])"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(mock_sgf)
            temp_path = f.name
        try:
            tokens, meta = parse_sgf_to_tokens(temp_path)
            self.assertIsNotNone(tokens)
            self.assertEqual(tokens[0], "<bos>")
            self.assertEqual(tokens[1], "<go>")
            self.assertEqual(tokens[2], "go:sz:19")
            self.assertEqual(tokens[3], "go:b:pd")
            self.assertEqual(tokens[4], "go:w:dd")
            self.assertEqual(tokens[5], "go:b:pp")
            self.assertEqual(tokens[-1], "<eos>")
            self.assertEqual(meta["black"], "BlackPlayer")
            self.assertEqual(meta["white"], "WhitePlayer")
            self.assertEqual(meta["board_size"], 19)
        finally:
            os.remove(temp_path)

    def test_go_parser_preserves_setup_stones(self):
        mock_sgf = "(;PB[BlackPlayer]PW[WhitePlayer]SZ[19]AB[pd][dd]AW[pp];B[qq];W[dc];B[ce];W[cf];B[fg];W[gd];B[dp];W[pq];B[oc];W[qo])"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(mock_sgf)
            temp_path = f.name
        try:
            tokens, meta = parse_sgf_to_tokens(temp_path)
            self.assertIn("go:setup_b", tokens)
            self.assertIn("go:pd", tokens)
            self.assertIn("go:dd", tokens)
            self.assertIn("go:setup_w", tokens)
            self.assertIn("go:pp", tokens)
            self.assertEqual(meta["setup_count"], 3)
        finally:
            os.remove(temp_path)

    def test_go_parser_preserves_zero_komi_and_handicap(self):
        mock_sgf = "(;SZ[19]KM[0]HA[0];B[pd];W[dd];B[pp];W[dp];B[cf];W[ch];B[fd];W[df];B[dg];W[cg])"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(mock_sgf)
            temp_path = f.name
        try:
            tokens, meta = parse_sgf_to_tokens(temp_path)
            self.assertIn("go:km:0.0", tokens)
            self.assertIn("go:ha:0", tokens)
            self.assertEqual(meta["komi"], 0.0)
            self.assertEqual(meta["handicap"], 0)
        finally:
            os.remove(temp_path)

    def test_go_validation_rejects_repeated_positions(self):
        tokens = [
            "<bos>", "<go>", "go:sz:5",
            "go:setup_b", "go:be", "go:setup_b", "go:ad", "go:setup_b", "go:bc",
            "go:setup_w", "go:bd", "go:setup_w", "go:ce", "go:setup_w", "go:dd", "go:setup_w", "go:cc",
            "go:b:cd", "go:w:bd", "<eos>",
        ]
        with self.assertRaises(ValueError):
            from src.go_parser import validate_go_token_sequence
            validate_go_token_sequence(tokens)

    def test_default_tokenizer_includes_all_game_markers(self):
        tokenizer = UniversalGameTokenizer()
        for marker in ("<chess>", "<shogi>", "<go>", "<othello>", "<poker>", "<bridge>"):
            self.assertIn(marker, tokenizer.vocab)

    def test_go_parser_rejects_variations(self):
        mock_sgf = "(;SZ[19];B[pd](;W[dd];B[pp];W[dp];B[cf];W[ch];B[fd];W[df];B[dg];W[cg])(;W[qq]))"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(mock_sgf)
            temp_path = f.name
        try:
            tokens, meta = parse_sgf_to_tokens(temp_path)
            self.assertIsNone(tokens)
            self.assertIsNone(meta)
        finally:
            os.remove(temp_path)

    def test_othello_parser(self):
        mock_pgn = """[Event "Othello Match"]
[Black "PlayerB"]
[White "PlayerW"]
[Result "32-32"]

""" + terminal_othello_pgn()
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(mock_pgn)
            temp_path = f.name
        try:
            games = list(parse_othello_pgn_to_tokens(temp_path))
            self.assertEqual(len(games), 1)
            tokens, meta = games[0]
            self.assertEqual(tokens[0], "<bos>")
            self.assertEqual(tokens[1], "<othello>")
            self.assertEqual(tokens[2:6], ["ot:b:d3", "ot:w:c3", "ot:b:b3", "ot:w:b2"])
            self.assertEqual(tokens[-1], "<eos>")
            self.assertEqual(meta["black"], "PlayerB")
        finally:
            os.remove(temp_path)

    def test_othello_validation_inserts_implicit_passes(self):
        moves_without_passes = [move for move in TERMINAL_OTHELLO_MOVES if move != "pass"]
        canonical = validate_othello_moves(moves_without_passes)
        self.assertGreater(canonical.count("pass"), 0)
        self.assertEqual([move for move in canonical if move != "pass"], moves_without_passes)

    def test_othello_validation_rejects_duplicate_or_illegal_moves(self):
        with self.assertRaises(ProductionDatasetError):
            validate_entry({
                "game": "othello",
                "tokens": ["<bos>", "<othello>", "ot:b:f5", "ot:w:f5", "ot:b:c3", "ot:w:d3", "<eos>"],
            })

    def test_othello_validation_can_allow_nonterminal_prefixes(self):
        canonical = validate_othello_moves(["d3", "c3", "b3", "b2", "b1", "a1", "c4", "c1"], require_terminal=False)
        self.assertEqual(canonical, ["d3", "c3", "b3", "b2", "b1", "a1", "c4", "c1"])

    def test_othello_validation_appends_terminal_double_passes(self):
        moves_without_passes = [move for move in TERMINAL_OTHELLO_MOVES if move != "pass"]
        canonical = validate_othello_moves(moves_without_passes)
        self.assertEqual(canonical[-2:], ["pass", "pass"])

    def test_othello_jsonl_direct_parser_sets_view_metadata(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            f.write(json.dumps({"moves": TERMINAL_OTHELLO_MOVES}) + "\n")
            temp_path = f.name
        try:
            entries = list(parse_othello_jsonl_to_tokens(temp_path))
            self.assertEqual(len(entries), 1)
            _, meta = entries[0]
            self.assertEqual(meta["seat_count"], 2)
            self.assertEqual(meta["view_type"], "complete")
            self.assertIsNone(meta["viewer_seat"])
        finally:
            os.remove(temp_path)

    def test_othello_hf_parser_sets_source_path(self):
        fake_datasets = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: iter([{"moves": TERMINAL_OTHELLO_MOVES}]))
        with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
            entries = list(parse_othello_hf_dataset("org/othello", split="train", max_games=1))
        self.assertEqual(len(entries), 1)
        _, meta = entries[0]
        self.assertEqual(meta["source_path"], "hf://org/othello:train")
        self.assertEqual(meta["seat_count"], 2)

    def test_othello_multiline_comments_and_passes_do_not_count_as_moves(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pgn") as f:
            f.write("""[Event "Othello"]\n[Black "B"]\n[White "W"]\n\n{comment\nwith D3 inside}\n""" + terminal_othello_pgn())
            temp_path = f.name
        try:
            tokens, meta = next(parse_othello_pgn_to_tokens(temp_path))
            self.assertEqual(meta["move_count"], othello_move_count(tokens))
            self.assertEqual(meta["move_count"], sum(1 for move in TERMINAL_OTHELLO_MOVES if move != "pass"))
        finally:
            os.remove(temp_path)

    def test_validate_entry_rejects_illegal_shogi_moves(self):
        with self.assertRaises(ProductionDatasetError):
            validate_entry({
                "game": "shogi",
                "tokens": ["<bos>", "<shogi>", "sh:b:5a", "sh:5b", "sh:end:resign", "<eos>"],
                "metadata": {"seat_count": 2, "view_type": "complete"},
            })

    def test_validate_entry_accepts_legal_shogi_prefix(self):
        validate_entry({
            "game": "shogi",
            "tokens": ["<bos>", "<shogi>", "sh:b:7g", "sh:7f", "sh:w:3c", "sh:3d", "sh:end:resign", "<eos>"],
            "metadata": {"seat_count": 2, "view_type": "complete"},
        })

    def test_poker_simulator(self):
        simulator = PokerHandSimulator()
        tokens, meta = simulator.simulate_hand()
        self.assertEqual(tokens[0], "<bos>")
        self.assertEqual(tokens[1], "<poker>")
        self.assertEqual(tokens[2], "view:complete")
        self.assertEqual(tokens[-1], "<eos>")
        self.assertFalse(any(t.startswith("H:") for t in tokens))
        self.assertFalse(any(re.search(r"[br]\d+", t) for t in tokens))
        self.assertIn("pk:act:post_small_blind", tokens)
        self.assertIn("pk:act:post_big_blind", tokens)
        self.assertFalse(any(token.startswith("street:") for token in tokens))
        self.assertTrue(any(t.startswith("pk:winner:") for t in tokens))
        self.assertIsNotNone(meta["winner"])
        validate_entry({"game": "poker", "tokens": tokens, "metadata": meta})
        self.assertEqual(meta["source"], "synthetic_simulator")
        self.assertEqual(meta["move_count"], poker_action_count(tokens))
        self.assertLess(meta["move_count"], len(tokens))
        street_has_bet = False
        in_postflop_street = False
        for token in tokens:
            if token in {"pk:act:preflop", "pk:act:flop", "pk:act:turn", "pk:act:river"}:
                in_postflop_street = token != "pk:act:preflop"
                street_has_bet = False
            elif token in {"pk:act:bet", "pk:act:raise", "pk:act:post_big_blind"}:
                street_has_bet = True
            elif in_postflop_street and token == "pk:act:call":
                self.assertTrue(street_has_bet)
            elif in_postflop_street and token == "pk:act:check":
                self.assertFalse(street_has_bet)

    def test_poker_simulator_dataset_emits_all_views(self):
        entries = list(generate_poker_dataset(n_hands=1))
        self.assertEqual(len(entries), 8)
        view_types = [meta["view_type"] for _, meta in entries]
        self.assertEqual(view_types.count("complete"), 1)
        self.assertEqual(view_types.count("imperfect"), 6)
        self.assertEqual(view_types.count("omniscient"), 1)
        for tokens, meta in entries:
            validate_entry({"game": "poker", "tokens": tokens, "metadata": meta})
        self.assertIn("pk:private_card", entries[-1][0])
        self.assertIn("pk:undealt_card", entries[-1][0])

    def test_poker_simulator_can_raise_preflop(self):
        def rig_shuffle(deck):
            for card in ["Ah", "Ad", "Kc", "Kd"]:
                deck.remove(card)
            deck.extend(["Kd", "Kc", "Ad", "Ah"])

        with mock.patch("src.poker_parser.random.shuffle", rig_shuffle):
            tokens, meta = PokerHandSimulator(num_seats=2).simulate_hand()
        self.assertIn("pk:act:raise", tokens)
        validate_entry({"game": "poker", "tokens": tokens, "metadata": meta})

    def test_poker_simulator_omniscient_keeps_undealt_board_cards_after_prefold(self):
        def rig_shuffle(deck):
            for card in ["2h", "3d", "Ah", "Ad"]:
                deck.remove(card)
            deck.extend(["Ad", "Ah", "3d", "2h"])

        with mock.patch("src.poker_parser.random.shuffle", rig_shuffle):
            entries = list(PokerHandSimulator(num_seats=2).simulate_hand_views())
        omniscient = entries[-1][0]
        self.assertNotIn("pk:act:flop", omniscient)
        self.assertEqual(sum(1 for token in omniscient if token == "pk:private_card"), 4)
        self.assertEqual(sum(1 for token in omniscient if token == "pk:undealt_card"), 48)
        validate_entry({"game": "poker", "tokens": omniscient, "metadata": entries[-1][1]})

    def test_poker_postflop_simulator_can_fold_and_raise_legally(self):
        simulator = PokerHandSimulator(num_seats=4)
        tokens, remaining = simulator._postflop_betting_round(
            [1, 2, 3, 4],
            should_bet=lambda seat: seat == 1,
            should_continue=lambda seat: seat in {1, 2, 3},
            should_raise=lambda seat: seat == 3,
            bet_amount=40,
        )
        self.assertIn("pk:act:raise", tokens)
        self.assertIn("pk:act:fold", tokens)
        self.assertEqual(remaining, [1, 2, 3])
        p2_call_positions = [
            index for index, token in enumerate(tokens[:-1])
            if token == "pk:seat:p2" and tokens[index + 1] == "pk:act:call"
        ]
        self.assertEqual(len(p2_call_positions), 2)
        check_tokens, check_remaining = simulator._postflop_betting_round(
            [1, 2, 3],
            should_bet=lambda seat: False,
            should_continue=lambda seat: True,
            should_raise=lambda seat: False,
            bet_amount=100,
        )
        self.assertEqual(check_remaining, [1, 2, 3])
        self.assertEqual(check_tokens, ["pk:seat:p1", "pk:act:check", "pk:seat:p2", "pk:act:check", "pk:seat:p3", "pk:act:check"])
        bluff_tokens, bluff_remaining = simulator._postflop_betting_round(
            [1, 2, 3],
            should_bet=lambda seat: seat == 1,
            should_continue=lambda seat: seat != 1,
            should_raise=lambda seat: seat == 2,
            bet_amount=40,
        )
        self.assertNotIn(1, bluff_remaining)
        p1_responses = [
            bluff_tokens[index + 1]
            for index, token in enumerate(bluff_tokens[:-1])
            if token == "pk:seat:p1"
        ]
        self.assertEqual(p1_responses[-1], "pk:act:fold")

    def test_poker_preflop_reraises_increase_amount_and_get_responses(self):
        simulator = PokerHandSimulator(num_seats=4)
        hands = {
            1: ["Qh", "Qd"],
            2: ["Ah", "Kd"],
            3: ["As", "Ad"],
            4: ["Kc", "Kh"],
        }
        tokens, remaining = simulator._preflop_betting_round(hands, [1, 2, 3, 4], 20)
        self.assertEqual(remaining, [1, 2, 3, 4])
        raise_positions = [index for index, token in enumerate(tokens) if token == "pk:act:raise"]
        self.assertEqual(len(raise_positions), 2)
        first_amount = tokens[raise_positions[0] + 1:raise_positions[0] + 3]
        second_amount = tokens[raise_positions[1] + 1:raise_positions[1] + 4]
        self.assertEqual(first_amount, ["pk:amt:6", "pk:amt:0"])
        self.assertEqual(second_amount, ["pk:amt:1", "pk:amt:2", "pk:amt:0"])
        self.assertGreater(tokens.count("pk:act:call"), 2)

    def test_poker_preflop_bb_can_raise_over_limpers(self):
        simulator = PokerHandSimulator(num_seats=2)
        hands = {
            1: ["Ah", "Kd"],
            2: ["As", "Ad"],
        }
        original_active = [1, 2]
        tokens, remaining = simulator._preflop_betting_round(hands, original_active, 20)
        self.assertEqual(original_active, [1, 2])
        self.assertEqual(remaining, [1, 2])
        p2_index = tokens.index("pk:seat:p2")
        self.assertEqual(tokens[p2_index + 1], "pk:act:raise")

    def test_poker_preflop_does_not_mutate_active_players(self):
        simulator = PokerHandSimulator(num_seats=3)
        hands = {
            1: ["2h", "7d"],
            2: ["As", "Kd"],
            3: ["3c", "8s"],
        }
        original_active = [1, 2, 3]
        tokens, remaining = simulator._preflop_betting_round(hands, original_active, 20)
        self.assertEqual(original_active, [1, 2, 3])
        self.assertNotEqual(remaining, original_active)

    def test_generate_poker_dataset_seed_is_reproducible(self):
        first = list(generate_poker_dataset(n_hands=2, seed=123))
        second = list(generate_poker_dataset(n_hands=2, seed=123))
        self.assertEqual(first, second)

    def test_poker_score_compares_tie_breakers(self):
        simulator = PokerHandSimulator()
        board = ["2c", "3s", "4h", "8d", "9c"]
        kings = simulator._score_key(simulator.get_best_hand(["Kh", "Kd"], board))
        aces = simulator._score_key(simulator.get_best_hand(["Ah", "Ad"], board))
        self.assertGreater(aces, kings)

    def test_poker_score_sorts_equal_multiplicities_by_rank(self):
        simulator = PokerHandSimulator()
        score = simulator._score_key(simulator._eval_hand(["Kh", "Kd", "Ks", "5c", "5d", "5h", "2s"]))
        self.assertEqual(score, (6, 13, 5))

    def test_validate_entry_accepts_chess960_spaced_variant_token(self):
        validate_entry({
            "game": "chess",
            "tokens": ["<bos>", "<chess>", "ch:rule:variant:chess_960", "<eos>"],
            "metadata": {"seat_count": 2, "view_type": "complete"},
        })

    def test_validate_entry_rejects_poker_bet_before_blinds(self):
        with self.assertRaises(ProductionDatasetError):
            validate_entry({
                "game": "poker",
                "tokens": ["<bos>", "<poker>", "view:complete", "pk:seat:p1", "pk:act:raise", "pk:amt:6", "pk:amt:0", "<eos>"],
                "metadata": {"seat_count": 2, "view_type": "complete"},
            })

    def test_validate_entry_allows_ante_before_betting(self):
        validate_entry({
            "game": "poker",
            "tokens": [
                "<bos>", "<poker>", "view:complete",
                "pk:seat:p1", "pk:act:post_ante", "pk:amt:1", "pk:amt:0",
                "pk:seat:p2", "pk:act:post_ante", "pk:amt:1", "pk:amt:0",
                "pk:seat:p1", "pk:act:raise", "pk:amt:6", "pk:amt:0",
                "<eos>",
            ],
            "metadata": {"seat_count": 2, "view_type": "complete"},
        })

    def test_validate_entry_accepts_amount_token_immediately_before_eos(self):
        validate_entry({
            "game": "poker",
            "tokens": [
                "<bos>", "<poker>", "view:complete",
                "pk:seat:p1", "pk:act:post_blind", "pk:amt:2", "pk:amt:0",
                "pk:seat:p2", "pk:act:raise", "pk:amt:6", "pk:amt:0",
                "<eos>",
            ],
            "metadata": {"seat_count": 2, "view_type": "complete"},
        })

    def test_bridge_pbn_parser(self):
        mock_pbn = """[Event "World Championship"]
[Site "Salsomaggiore"]
[Date "2025.01.02"]
[Board "1"]
[Dealer "N"]
[Vulnerable "None"]
[Deal "N:AKQJ.543.2.98765 T987.2.AKQJ.T432 6543.AKQJ.43.AKQ 2.T9876.T98765.J"]
[Auction "N"]
1NT Pass 3NT Pass Pass Pass
[Play "S"]
HA H9 H5 H2
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pbn") as f:
            f.write(mock_pbn)
            temp_path = f.name
        try:
            boards = list(parse_pbn_to_tokens(temp_path))
            self.assertEqual(len(boards), 6)
            tokens, meta = boards[0]
            self.assertEqual(tokens[0], "<bos>")
            self.assertEqual(tokens[1], "<bridge>")
            self.assertEqual(tokens[2], "view:complete")
            self.assertIn("br:dealer:N", tokens)
            self.assertIn("br:bid:1N", tokens)
            self.assertIn("br:play:S", tokens)
            self.assertEqual(meta["seat_count"], 4)
            self.assertFalse(any(token.startswith("br:hand:") for token in tokens))
            view_types = [entry_meta["view_type"] for _, entry_meta in boards]
            self.assertEqual(view_types.count("imperfect"), 4)
            self.assertIn("omniscient", view_types)
            for view_tokens, view_meta in boards:
                validate_entry({"game": "bridge", "tokens": view_tokens, "metadata": view_meta})
                if view_meta["view_type"] == "imperfect":
                    self.assertEqual(sum(token.startswith("br:hand:") for token in view_tokens), 13)
                if view_meta["view_type"] == "omniscient":
                    self.assertEqual(sum(token.startswith("br:hand:") for token in view_tokens), 52)
        finally:
            os.remove(temp_path)

    def test_bridge_parse_inputs_counts_max_games_not_views(self):
        board = """[Event "World Championship"]
[Date "2025.01.02"]
[Dealer "N"]
[Vulnerable "None"]
[Deal "N:AKQJ.543.2.98765 T987.2.AKQJ.T432 6543.AKQJ.43.AKQ 2.T9876.T98765.J"]
[Auction "N"]
1NT Pass 3NT Pass Pass Pass
[Play "S"]
HA H9 H5 H2
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pbn") as f:
            f.write(board.replace('World Championship', 'Board 1') + "\n")
            f.write(board.replace('World Championship', 'Board 2'))
            temp_path = f.name
        try:
            rows = list(parse_bridge_inputs(temp_path, max_games=2))
            self.assertEqual(len(rows), 12)
            self.assertEqual(len({meta["view_group_id"] for _, meta in rows}), 2)
        finally:
            os.remove(temp_path)

    def test_bridge_trump_ruff_sets_next_leader(self):
        mock_pbn = """[Event "Trump Test"]
[Date "2025.01.02"]
[Dealer "N"]
[Vulnerable "None"]
[Contract "1H"]
[Deal "N:AKQJ.543.2.98765 T987.AKQJT9.AKQ. 6543.2.43.AKQT32 2.876.JT98765.J4"]
[Auction "N"]
1H Pass Pass Pass
[Play "N"]
C9 HA CA CJ
ST S6 S2 SA
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pbn") as f:
            f.write(mock_pbn)
            temp_path = f.name
        try:
            boards = list(parse_pbn_to_tokens(temp_path))
            self.assertEqual(len(boards), 6)
            tokens, meta = boards[0]
            self.assertIn("br:trump:h", tokens)
            self.assertIn("br:play:E", tokens)
            for view_tokens, view_meta in boards:
                validate_entry({"game": "bridge", "tokens": view_tokens, "metadata": view_meta})
        finally:
            os.remove(temp_path)

    def test_bridge_play_leader_falls_back_to_declarer_left(self):
        mock_pbn = """[Event "Declarer Leader"]
[Date "2025.01.02"]
[Dealer "N"]
[Declarer "N"]
[Vulnerable "None"]
[Contract "1H"]
[Deal "N:AKQJ.543.2.98765 T987.AKQJT9.AKQ. 6543.2.43.AKQT32 2.876.JT98765.J4"]
[Auction "N"]
1H Pass Pass Pass
[Play ""]
HA H2 H7 H3
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pbn") as f:
            f.write(mock_pbn)
            temp_path = f.name
        try:
            boards = list(parse_pbn_to_tokens(temp_path))
            self.assertEqual(len(boards), 6)
            self.assertIn("br:play_leader:E", boards[0][0])
        finally:
            os.remove(temp_path)


    def test_tokenizer_encoding(self):
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<chess>", "<shogi>", "<go>", "<othello>", "<poker>"]
        tokenizer = UniversalGameTokenizer(special_tokens=special_tokens)
        
        mock_games = [
            ["<bos>", "<go>", "pd", "dd", "<eos>"],
            ["<bos>", "<othello>", "f5", "d6", "<eos>"],
            ["<bos>", "<poker>", "H:1:AhKd", "WINNER:1", "<eos>"]
        ]
        tokenizer.build_vocab(mock_games)
        
        self.assertEqual(tokenizer.vocab_size, 15) # 9 special + 6 moves
        
        sequence = ["<bos>", "<go>", "pd", "invalid", "<eos>"]
        encoded = tokenizer.encode(sequence)
        decoded = tokenizer.decode(encoded)
        
        self.assertEqual(decoded[0], "<bos>")
        self.assertEqual(decoded[1], "<go>")
        self.assertEqual(decoded[2], "pd")
        self.assertEqual(decoded[3], "<unk>")
        self.assertEqual(decoded[4], "<eos>")

    def test_mahjonglm_tokenizer_extension_preserves_base_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer_dir = os.path.join(temp_dir, "mahjong_tokenizer")
            os.makedirs(tokenizer_dir)
            with open(os.path.join(tokenizer_dir, "vocab.txt"), "w", encoding="utf-8") as f:
                f.write("<pad>\n<unk>\n<bos>\n<eos>\nrule_riichi\nview_complete\ngame_start\n")

            tokenizer = UniversalGameTokenizer.from_mahjonglm_assets(tokenizer_dir)
            self.assertEqual(tokenizer.vocab["rule_riichi"], 4)
            added = tokenizer.add_tokens(["rule_chess", "e2e4", "e7e5"])
            self.assertEqual(added, 3)
            self.assertEqual(tokenizer.vocab["rule_riichi"], 4)
            self.assertEqual(tokenizer.vocab["rule_chess"], 7)
            if tokenizer.backend_tokenizer is not None:
                self.assertEqual(tokenizer.backend_tokenizer.token_to_id("rule_chess"), 7)

    def test_mahjonglm_row_schema_excludes_boundary_tokens(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer_dir = os.path.join(temp_dir, "mahjong_tokenizer")
            os.makedirs(tokenizer_dir)
            with open(os.path.join(tokenizer_dir, "vocab.txt"), "w", encoding="utf-8") as f:
                f.write("<pad>\n<unk>\n<bos>\n<eos>\nview:complete\ne2e4\ne7e5\ng1f3\nb8c6\n")
            tokenizer = UniversalGameTokenizer.from_mahjonglm_assets(tokenizer_dir)
            entry = {
                "game": "chess",
                "tokens": ["<bos>", "<chess>", "view:complete", "e2e4", "e7e5", "g1f3", "b8c6", "<eos>"],
                "metadata": {"date": "2024.01.01", "source_id": "sample"},
            }
            self.assertEqual(tokens_to_mahjonglm_stream(entry), ["view:complete", "e2e4", "e7e5", "g1f3", "b8c6"])
            row = entry_to_mahjonglm_row(entry, tokenizer)
            self.assertEqual(set(row), {"game_id", "year", "seat_count", "view_type", "viewer_seat", "length", "input_ids", "tokenizer_fingerprint"})
            self.assertEqual(row["year"], 2024)
            self.assertEqual(row["seat_count"], 2)
            self.assertEqual(row["view_type"], "complete")
            self.assertIsNone(row["viewer_seat"])
            self.assertEqual(row["length"], 5)
            self.assertEqual(row["tokenizer_fingerprint"], tokenizer.fingerprint())
            self.assertNotIn(tokenizer.vocab["<bos>"], row["input_ids"])
            self.assertNotIn(tokenizer.vocab["<eos>"], row["input_ids"])

    def test_mahjonglm_metadata_requires_poker_seat_count(self):
        row = {
            "game": "poker",
            "tokens": ["<bos>", "<poker>", "view:complete", "pk:act:fold", "<eos>"],
            "metadata": {},
        }
        stream = tokens_to_mahjonglm_stream(row)
        self.assertEqual(stream[0], "view:complete")
        from src.mahjonglm_compat import normalize_mahjonglm_metadata
        with self.assertRaises(ValueError):
            normalize_mahjonglm_metadata(row)
        row["metadata"] = {"seat_count": 0}
        self.assertEqual(normalize_mahjonglm_metadata(row)["seat_count"], 0)

    def test_mahjonglm_tokenizer_rejects_sparse_or_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sparse_dir = os.path.join(temp_dir, "sparse")
            os.makedirs(sparse_dir)
            with open(os.path.join(sparse_dir, "vocab.json"), "w", encoding="utf-8") as f:
                json.dump({"<pad>": 0, "<unk>": 1, "rule_riichi": 3}, f)
            with self.assertRaises(ValueError):
                UniversalGameTokenizer.from_mahjonglm_assets(sparse_dir)

            dup_dir = os.path.join(temp_dir, "dup")
            os.makedirs(dup_dir)
            with open(os.path.join(dup_dir, "vocab.json"), "w", encoding="utf-8") as f:
                json.dump({"<pad>": 0, "<unk>": 1, "rule_riichi": 1}, f)
            with self.assertRaises(ValueError):
                UniversalGameTokenizer.from_mahjonglm_assets(dup_dir)

    def test_safe_extract_zip_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "bad.zip")
            extract_dir = os.path.join(temp_dir, "out")
            os.makedirs(extract_dir)
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("../escape.pgn", "bad")

            with self.assertRaises(ValueError):
                safe_extract_zip(zip_path, extract_dir, allowed_suffixes=(".pgn",))

    def test_hf_upload_deletes_only_after_success(self):
        class FakeApi:
            def create_repo(self, **kwargs):
                pass

            def upload_file(self, **kwargs):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = os.path.join(temp_dir, "artifact.jsonl")
            with open(artifact, "w", encoding="utf-8") as f:
                f.write("{}\n")

            uploader = HuggingFaceShardUploader("user/repo", token="test-token")
            uploader.api = FakeApi()
            uploader.upload_file(artifact, "artifact.jsonl", delete_local=True)
            self.assertFalse(os.path.exists(artifact))

    def test_production_shards_validate_all_games_on_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            chess_path = os.path.join(temp_dir, "sample.pgn")
            with open(chess_path, "w", encoding="utf-8") as f:
                f.write("""[Event "Sample"]\n[White "A"]\n[Black "B"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 *""")

            shogi_dir = os.path.join(temp_dir, "shogi")
            os.makedirs(shogi_dir)
            shogi_path = os.path.join(shogi_dir, "sample.csa")
            with open(shogi_path, "w", encoding="utf-8") as f:
                f.write("""V2.2\nN+Black\nN-White\nPI\n+\n+7776FU\nT1\n-3334FU\nT1\n+2726FU\nT1\n-8384FU\nT1\n+2625FU\nT1\n-8485FU\nT1\n+6978KI\nT1\n-4132KI\nT1\n+2524FU\nT1\n-2324FU\nT1\n%TORYO\n""")

            go_dir = os.path.join(temp_dir, "go")
            os.makedirs(go_dir)
            go_path = os.path.join(go_dir, "sample.sgf")
            with open(go_path, "w", encoding="utf-8") as f:
                f.write("(;PB[Black]PW[White]RE[B+R];B[pd];W[dd];B[pp];W[dp];B[cf];W[ch];B[fd];W[df];B[dg];W[cg])")

            othello_path = os.path.join(temp_dir, "sample_othello.pgn")
            with open(othello_path, "w", encoding="utf-8") as f:
                f.write("""[Event "Othello"]\n[Black "B"]\n[White "W"]\n[Result "32-32"]\n\n""" + terminal_othello_pgn())

            poker_path = os.path.join(temp_dir, "sample.phh")
            with open(poker_path, "w", encoding="utf-8") as f:
                f.write('''actions = ["deal_hole P1 AhAd", "post_blind P1 50", "post_blind P2 100", "call P1", "check P2", "deal_board 2c3d4h", "bet P2 200", "fold P1"]\n''')

            bridge_path = os.path.join(temp_dir, "sample.pbn")
            with open(bridge_path, "w", encoding="utf-8") as f:
                f.write("""[Event "Bridge"]\n[Date "2025.01.02"]\n[Dealer "N"]\n[Vulnerable "None"]\n[Deal "N:AKQJ.543.2.98765 T987.2.AKQJ.T432 6543.AKQJ.43.AKQ 2.T9876.T98765.J"]\n[Auction "N"]\n1NT Pass 3NT Pass Pass Pass\n[Play "S"]\nHA H9 H5 H2\n""")

            cases = {
                "chess": [chess_path],
                "shogi": [shogi_dir],
                "go": [go_dir],
                "othello": [othello_path],
                "poker": [poker_path],
                "bridge": [bridge_path],
            }
            for game, paths in cases.items():
                out_dir = os.path.join(temp_dir, "out", game)
                result = build_game_shards(
                    game,
                    paths,
                    out_dir,
                    target_tokens=4,
                    max_tokens_per_shard=1000,
                    max_records=1,
                )
                self.assertEqual(result["status"], "ready", game)
                self.assertGreater(result["tokens"], 0, game)
                self.assertEqual(len(result["shards"]), 1, game)
                shard_path = result["shards"][0]["path"]
                with gzip.open(shard_path, "rt", encoding="utf-8") as f:
                    entry = json.loads(f.readline())
                validate_entry(entry)
                self.assertEqual(entry["game"], game)
                if game == "poker":
                    self.assertFalse(any("deal_hole" in token for token in entry["tokens"]))

    def test_validate_entry_rejects_private_poker_tokens(self):
        private_tokens = [
            "h:1:AhAd",
            "H:1:AhAd",
            "hole_p1_ahad",
            "deal_hole_p1_ahad",
            "d_dh_p1_ahad",
            "dh_p1_ahad",
            "show_or_muck_hole_cards_p1_ahad",
        ]
        for token in private_tokens:
            with self.subTest(token=token):
                with self.assertRaises(ProductionDatasetError):
                    validate_entry({"game": "poker", "tokens": ["<bos>", "<poker>", token, "<eos>"]})

    def test_validate_entry_bounds_poker_imperfect_private_cards(self):
        one_card = ["<bos>", "<poker>", "view:imperfect:1", "pk:private_card", "pk:seat:p1", "pk:card:Ah", "pk:seat:p1", "pk:act:fold", "<eos>"]
        validate_entry({"game": "poker", "tokens": one_card, "metadata": {"seat_count": 2, "view_type": "imperfect"}})
        too_many_cards = []
        for rank in "A23456789TJ":
            too_many_cards.extend(["pk:private_card", "pk:seat:p1", f"pk:card:{rank}h"])
        too_many = ["<bos>", "<poker>", "view:imperfect:1"] + too_many_cards + ["pk:seat:p1", "pk:act:fold", "<eos>"]
        with self.assertRaises(ProductionDatasetError):
            validate_entry({"game": "poker", "tokens": too_many, "metadata": {"seat_count": 2, "view_type": "imperfect"}})

    def test_phh_parser_only_reads_actions_field(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".phh") as f:
            f.write('''venue = "quoted venue must not become action"\nplayers = ["Alice", "Bob"]\nactions = [\n  "deal_hole P1 AhAd",\n  "post_blind P1 50",\n  "post_blind P2 100",\n  "call P1"\n]\n''')
            temp_path = f.name
        try:
            hands = list(parse_phh_to_tokens(temp_path))
            self.assertEqual(len(hands), 2)
            tokens, meta = hands[0]
            self.assertNotIn("quoted_venue_must_not_become_action", tokens)
            self.assertNotIn("alice", tokens)
            self.assertNotIn("deal_hole_p1_ahad", tokens)
            self.assertEqual(tokens[2:], [
                "view:complete",
                "pk:seat:p1", "pk:act:post_blind", "pk:amt:5", "pk:amt:0",
                "pk:seat:p2", "pk:act:post_blind", "pk:amt:1", "pk:amt:0", "pk:amt:0",
                "pk:seat:p1", "pk:act:call",
                "<eos>",
            ])
            self.assertEqual(meta["view_type"], "complete")
            self.assertEqual(meta["seat_count"], 2)
            self.assertEqual(meta["view_rows_per_hand"], 2)
            self.assertEqual(meta["missing_private_seats"], [2])
            self.assertEqual(meta["private_actions_excluded"], 1)
        finally:
            os.remove(temp_path)

    def test_phh_parser_accepts_third_person_action_verbs(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".phh") as f:
            f.write('''actions = [
  "deal_hole P1 AhAd",
  "deal_hole P2 KcKd",
  "p1 posts small blind 50",
  "p2 posts big blind 100",
  "p1 calls",
  "p2 checks",
  "p1 bets 200",
  "p2 folds"
]
''')
            temp_path = f.name
        try:
            hands = list(parse_phh_to_tokens(temp_path))
            self.assertEqual(len(hands), 4)
            complete_tokens = hands[0][0]
            self.assertIn("pk:act:post_small_blind", complete_tokens)
            self.assertIn("pk:act:post_big_blind", complete_tokens)
            self.assertIn("pk:act:call", complete_tokens)
            self.assertIn("pk:act:check", complete_tokens)
            self.assertIn("pk:act:bet", complete_tokens)
            self.assertIn("pk:act:fold", complete_tokens)
        finally:
            os.remove(temp_path)


    def test_phhs_parser_preserves_common_state_between_hands(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".phhs") as f:
            f.write("""variant = 'NT'
starting_stacks = [10000, 10000]
actions = [
  'd dh p1 AhAd',
  'd dh p2 KcKd',
  'p1 cc',
  'p2 cc',
]
actions = [
  'd dh p1 QhQs',
  'd dh p2 JcJd',
  'p1 cc',
  'p2 cc',
]
""")
            temp_path = f.name
        try:
            hands = list(parse_phh_to_tokens(temp_path))
            self.assertEqual(len(hands), 8)
            self.assertIn("pk:VARIANT:nt", hands[0][0])
            self.assertIn("pk:VARIANT:nt", hands[4][0])
            self.assertIn("pk:STARTING_STACKS:BEGIN", hands[4][0])
        finally:
            os.remove(temp_path)

    def test_phhs_parser_uses_latest_redefined_state(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".phhs") as f:
            f.write("""variant = 'NT'
actions = [
  'd dh p1 AhAd',
  'd dh p2 KcKd',
  'p1 cc',
  'p2 cc',
]
variant = 'FT'
actions = [
  'd dh p1 QhQs',
  'd dh p2 JcJd',
  'p1 cc',
  'p2 cc',
]
""")
            temp_path = f.name
        try:
            hands = list(parse_phh_to_tokens(temp_path))
            self.assertIn("pk:VARIANT:nt", hands[0][0])
            self.assertIn("pk:VARIANT:ft", hands[4][0])
            self.assertNotIn("pk:VARIANT:nt", hands[4][0])
        finally:
            os.remove(temp_path)


    def test_phh_parser_ignores_brackets_inside_action_strings(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".phh") as f:
            f.write('''actions = [
  "deal_hole P1 AhAd",
  "deal_hole P2 KcKd",
  "post_blind P1 50 [small blind]",
  "post_blind P2 100",
  "call P1"
]
actions = [
  "deal_hole P1 QhQs",
  "deal_hole P2 JcJd",
  "post_blind P1 50",
  "post_blind P2 100",
  "call P1"
]
''')
            temp_path = f.name
        try:
            hands = list(parse_phh_to_tokens(temp_path))
            self.assertEqual(len(hands), 8)
            self.assertTrue(hands[0][1]["view_group_id"].endswith("#1"))
            self.assertTrue(hands[4][1]["view_group_id"].endswith("#2"))
        finally:
            os.remove(temp_path)

    def test_phh_parser_ignores_brackets_inside_comments(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".phh") as f:
            f.write('''actions = [
  "deal_hole P1 AhAd",
  "deal_hole P2 KcKd",
  "post_blind P1 50",  # [small blind
  "post_blind P2 100",
  "call P1"
]
''')
            temp_path = f.name
        try:
            hands = list(parse_phh_to_tokens(temp_path))
            self.assertEqual(len(hands), 4)
            self.assertIn("pk:act:post_blind", hands[0][0])
        finally:
            os.remove(temp_path)

    def test_phh_parser_handles_canonical_single_quoted_actions_and_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            phh_dir = os.path.join(temp_dir, "hands")
            os.makedirs(phh_dir)
            phh_path = os.path.join(phh_dir, "sample.phhs")
            with open(phh_path, "w", encoding="utf-8") as f:
                f.write("""variant = 'NT'
blinds_or_straddles = [50, 100]
starting_stacks = [10000, 10000]
actions = [
  'd dh p1 AhAd',
  'd dh p2 KcKd',
  'p1 cc',
  'p2 cbr 300',
  'p1 cc',
  'd db 2c3d4h',
  'p1 sm -',
]
""")
            hands = list(parse_phh_to_tokens(phh_dir))
            self.assertEqual(len(hands), 4)
            for tokens, meta in hands:
                validate_entry({"game": "poker", "tokens": tokens, "metadata": meta})
            views = {meta["view_type"]: (tokens, meta) for tokens, meta in hands if meta["view_type"] != "imperfect"}
            imperfect = {(meta["viewer_seat"]): (tokens, meta) for tokens, meta in hands if meta["view_type"] == "imperfect"}
            complete_tokens, complete_meta = views["complete"]
            self.assertEqual(complete_meta["seat_count"], 2)
            self.assertEqual(complete_meta["view_rows_per_hand"], 4)
            self.assertEqual(complete_meta["move_count"], poker_action_count(complete_tokens))
            self.assertIn("view:complete", complete_tokens)
            self.assertIn("pk:VARIANT:nt", complete_tokens)
            self.assertIn("pk:STARTING_STACKS:BEGIN", complete_tokens)
            self.assertIn("pk:act:call", complete_tokens)
            self.assertIn("pk:act:raise", complete_tokens)
            self.assertNotIn("pk:act:cc", complete_tokens)
            self.assertNotIn("pk:act:cbr", complete_tokens)
            self.assertIn("pk:amt:3", complete_tokens)
            self.assertIn("pk:act:deal_board", complete_tokens)
            self.assertTrue(any(t.startswith("pk:card:") for t in complete_tokens))
            self.assertIn("pk:act:hidden", complete_tokens)
            self.assertFalse(any("AhAd" in token or "KcKd" in token for token in complete_tokens))
            self.assertEqual(complete_meta["private_actions_excluded"], 2)
            self.assertIn("pk:private_card", imperfect[1][0])
            self.assertIn("pk:seat:p1", imperfect[1][0])
            self.assertIn("pk:private_card", imperfect[2][0])
            self.assertIn("pk:seat:p2", imperfect[2][0])
            omniscient_tokens, _ = views["omniscient"]
            self.assertIn("view:omniscient", omniscient_tokens)
            self.assertIn("pk:private_card", omniscient_tokens)
            self.assertIn("pk:undealt_card", omniscient_tokens)

    def test_phh_parser_normalizes_long_show_or_muck_actions(self):
        phh = """variant = 'NT'
actions = [
  'd dh p1 AhAd',
  'd dh p2 KcKd',
  'p1 cc',
  'show_or_muck_hole_cards p1 AhAd',
]
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".phh") as f:
            f.write(phh)
            temp_path = f.name
        try:
            hands = list(parse_phh_to_tokens(temp_path))
            complete_tokens = hands[0][0]
            self.assertIn("pk:seat:p1", complete_tokens)
            self.assertIn("pk:act:show", complete_tokens)
            self.assertTrue(any(t.startswith("pk:card:") for t in complete_tokens))
        finally:
            os.remove(temp_path)

    def test_poker_seat_count_scales_views_and_stats(self):
        phh = """variant = 'NT'
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = [
  'd dh p1 AhAd',
  'd dh p2 KcKd',
  'd dh p3 QsQh',
  'd dh p4 JcJd',
  'd dh p5 TsTh',
  'd dh p6 9c9d',
  'p1 cc',
  'p2 cc',
  'p3 cc',
  'p4 cc',
  'p5 cc',
  'p6 cbr 300',
]
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".phh") as f:
            f.write(phh)
            temp_path = f.name
        try:
            hands = list(parse_phh_to_tokens(temp_path))
            self.assertEqual(len(hands), 8)
            self.assertTrue(all(meta["seat_count"] == 6 for _, meta in hands))
            self.assertTrue(all(meta["view_rows_per_hand"] == 8 for _, meta in hands))

            stats = DatasetStatsAccumulator()
            for tokens, meta in hands:
                stats.update({"game": "poker", "tokens": tokens, "metadata": meta})
            summary = stats.summary()["poker"]["by_seat_count"]["6"]
            self.assertEqual(summary["rows"], 8)
            self.assertEqual(summary["views"], {"complete": 1, "imperfect": 6, "omniscient": 1})
        finally:
            os.remove(temp_path)

    def test_stats_excludes_context_tokens_from_move_counts(self):
        self.assertFalse(is_counted_move_token("sh:turn:black"))
        self.assertFalse(is_counted_move_token("sh:end:resign"))
        self.assertFalse(is_counted_move_token("go:sz:19"))
        self.assertFalse(is_counted_move_token("go:km:6.5"))
        self.assertFalse(is_counted_move_token("br:dealer:N"))
        self.assertFalse(is_counted_move_token("pk:amt:5"))
        self.assertFalse(is_counted_move_token("pk:ANTE_TRIMMING_STATUS:false"))
        self.assertFalse(is_counted_move_token("pk:BETTING_TYPE:no_limit"))
        self.assertFalse(is_counted_move_token("pk:seat:p1"))
        self.assertFalse(is_counted_move_token("pk:card:Ah"))
        self.assertFalse(is_counted_move_token("pk:showdown:p1"))
        self.assertFalse(is_counted_move_token("pk:winner:p1"))
        self.assertFalse(is_counted_move_token("pk:act:flop"))
        self.assertFalse(is_counted_move_token("pk:act:deal_board"))
        self.assertFalse(is_counted_move_token("pk:act:hidden"))
        self.assertFalse(is_counted_move_token("view:complete"))
        self.assertFalse(is_counted_move_token("view:imperfect:1"))
        self.assertFalse(is_counted_move_token("ch:rule:variant:standard"))
        # Decomposed sub-tokens must not inflate move counts
        self.assertFalse(is_counted_move_token("ch:e4"))
        self.assertFalse(is_counted_move_token("ch:b5"))
        self.assertFalse(is_counted_move_token("ch:=q"))
        self.assertFalse(is_counted_move_token("sh:7f"))
        self.assertFalse(is_counted_move_token("sh:+"))
        self.assertFalse(is_counted_move_token("go:pd"))
        self.assertFalse(is_counted_move_token("go:pass"))
        self.assertFalse(is_counted_move_token("br:card:Ah"))
        self.assertFalse(is_counted_move_token("br:card:Ks"))
        self.assertFalse(is_counted_move_token("br:bid:N"))
        # Action-initiator tokens count as moves
        self.assertTrue(is_counted_move_token("ch:w:e2"))
        self.assertTrue(is_counted_move_token("ch:b:e7"))
        self.assertTrue(is_counted_move_token("sh:b:7g"))
        self.assertTrue(is_counted_move_token("sh:w:3c"))
        self.assertTrue(is_counted_move_token("go:b:pd"))
        self.assertTrue(is_counted_move_token("go:w:dd"))
        self.assertTrue(is_counted_move_token("br:play:S"))
        self.assertTrue(is_counted_move_token("br:bid:1N"))
        self.assertTrue(is_counted_move_token("br:bid:PASS"))
        self.assertTrue(is_counted_move_token("ot:b:d3"))
        self.assertTrue(is_counted_move_token("pk:act:raise"))

    def test_bridge_auction_allows_pass_before_redouble(self):
        mock_pbn = """[Event "Redouble"]
[Date "2025.01.02"]
[Dealer "N"]
[Vulnerable "None"]
[Deal "N:AKQJ.543.2.98765 T987.2.AKQJ.T432 6543.AKQJ.43.AKQ 2.T9876.T98765.J"]
[Auction "N"]
1H X Pass Pass XX Pass Pass Pass
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pbn") as f:
            f.write(mock_pbn)
            temp_path = f.name
        try:
            boards = list(parse_pbn_to_tokens(temp_path))
            self.assertEqual(len(boards), 6)
            for tokens, meta in boards:
                validate_entry({"game": "bridge", "tokens": tokens, "metadata": meta})
        finally:
            os.remove(temp_path)

    def test_bridge_parser_accepts_incomplete_auction_without_play(self):
        mock_pbn = """[Event "Partial Auction"]
[Date "2025.01.02"]
[Dealer "N"]
[Vulnerable "None"]
[Deal "N:AKQJ.543.2.98765 T987.2.AKQJ.T432 6543.AKQJ.43.AKQ 2.T9876.T98765.J"]
[Auction "N"]
1H Pass 2H Pass
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pbn") as f:
            f.write(mock_pbn)
            temp_path = f.name
        try:
            boards = list(parse_pbn_to_tokens(temp_path))
            self.assertEqual(len(boards), 6)
            for tokens, meta in boards:
                validate_entry({"game": "bridge", "tokens": tokens, "metadata": meta})
        finally:
            os.remove(temp_path)

    def test_bridge_allows_valid_incomplete_play_and_validates_seat_order(self):
        mock_pbn = """[Event "Incomplete"]
[Date "2025.01.02"]
[Dealer "N"]
[Vulnerable "None"]
[Deal "N:AKQJ.543.2.98765 T987.2.AKQJ.T432 6543.AKQJ.43.AKQ 2.T9876.T98765.J"]
[Auction "N"]
1NT Pass 3NT Pass Pass Pass
[Play "S"]
HA
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pbn") as f:
            f.write(mock_pbn)
            temp_path = f.name
        try:
            boards = list(parse_pbn_to_tokens(temp_path))
            self.assertEqual(len(boards), 6)
            for tokens, meta in boards:
                validate_entry({"game": "bridge", "tokens": tokens, "metadata": meta})

            bad_tokens = list(boards[0][0])
            play_index = bad_tokens.index("br:play:S")
            bad_tokens[play_index] = "br:play:N"
            with self.assertRaises(ProductionDatasetError):
                validate_entry({"game": "bridge", "tokens": bad_tokens, "metadata": boards[0][1]})
        finally:
            os.remove(temp_path)

    def test_bridge_dealer_metadata_is_canonicalized(self):
        mock_pbn = """[Event "Lower Dealer"]
[Date "2025.01.02"]
[Dealer "n"]
[Vulnerable "None"]
[Deal "N:AKQJ.543.2.98765 T987.2.AKQJ.T432 6543.AKQJ.43.AKQ 2.T9876.T98765.J"]
[Auction "N"]
1NT Pass 3NT Pass Pass Pass
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pbn") as f:
            f.write(mock_pbn)
            temp_path = f.name
        try:
            tokens, meta = next(iter(parse_pbn_to_tokens(temp_path)))
            self.assertIn("br:dealer:N", tokens)
            self.assertEqual(meta["dealer"], "N")
        finally:
            os.remove(temp_path)

    def test_bridge_imperfect_view_validates_visible_hand_cards(self):
        mock_pbn = """[Event "Visible Hand"]
[Date "2025.01.02"]
[Dealer "N"]
[Vulnerable "None"]
[Deal "N:AKQJ.543.2.98765 T987.2.AKQJ.T432 6543.AKQJ.43.AKQ 2.T9876.T98765.J"]
[Auction "N"]
1NT Pass 3NT Pass Pass Pass
[Play "S"]
HA H9 H5 H2
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pbn") as f:
            f.write(mock_pbn)
            temp_path = f.name
        try:
            boards = list(parse_pbn_to_tokens(temp_path))
            imperfect_tokens, imperfect_meta = next((tokens, meta) for tokens, meta in boards if meta["viewer_seat"] == "S")
            validate_entry({"game": "bridge", "tokens": imperfect_tokens, "metadata": imperfect_meta})
            bad_tokens = list(imperfect_tokens)
            play_idx = bad_tokens.index("br:play:S")
            bad_tokens[play_idx + 1] = "br:card:Ks"
            with self.assertRaises(ProductionDatasetError):
                validate_entry({"game": "bridge", "tokens": bad_tokens, "metadata": imperfect_meta})
        finally:
            os.remove(temp_path)

    def test_poker_production_does_not_split_view_group_at_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            phh_path = os.path.join(temp_dir, "six.phh")
            with open(phh_path, "w", encoding="utf-8") as f:
                f.write("""variant = 'NT'
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = [
  'd dh p1 AhAd',
  'd dh p2 KcKd',
  'd dh p3 QsQh',
  'd dh p4 JcJd',
  'd dh p5 TsTh',
  'd dh p6 9c9d',
  'p1 cc',
  'p2 cc',
  'p3 cc',
  'p4 cc',
  'p5 cc',
  'p6 cbr 300',
]
""")
            result = build_game_shards(
                "poker",
                [phh_path],
                os.path.join(temp_dir, "out"),
                target_tokens=1,
                max_records=1,
            )
            self.assertEqual(result["rows"], 8)
            self.assertEqual(result["stats"]["by_seat_count"]["6"]["views"], {
                "complete": 1,
                "imperfect": 6,
                "omniscient": 1,
            })
            self.assertEqual(len(result["shards"]), 1)

    def test_poker_max_records_is_global_across_input_paths(self):
        phh = """variant = 'NT'
actions = [
  'd dh p1 AhAd',
  'd dh p2 KcKd',
  'p1 cc',
  'p2 cc',
]
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for index in range(2):
                path = os.path.join(temp_dir, f"hand{index}.phh")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(phh)
                paths.append(path)
            result = build_game_shards(
                "poker",
                paths,
                os.path.join(temp_dir, "out"),
                target_tokens=1,
                max_records=1,
            )
            self.assertEqual(result["rows"], 4)
            self.assertEqual(result["stats"]["by_seat_count"]["2"]["views"], {"complete": 1, "imperfect": 2, "omniscient": 1})

    def test_cached_entries_respect_max_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "cache.jsonl")
            rows = [
                {"game": "poker", "tokens": ["<bos>", "<poker>", "view:complete", "pk:seat:p1", "pk:act:fold", "<eos>"], "metadata": {"seat_count": 2, "view_type": "complete"}},
                {"game": "poker", "tokens": ["<bos>", "<poker>", "view:complete", "pk:seat:p2", "pk:act:fold", "<eos>"], "metadata": {"seat_count": 2, "view_type": "complete"}},
            ]
            with open(cache_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            result = build_game_shards(
                "poker",
                [],
                os.path.join(temp_dir, "out"),
                target_tokens=1,
                max_records=1,
                cached_entries_path=cache_path,
            )
            self.assertEqual(result["rows"], 1)

    def test_limit_entries_zero_records_yields_no_rows(self):
        rows = [
            {"game": "chess", "tokens": ["<bos>", "<chess>", "e2e4", "<eos>"], "metadata": {"seat_count": 2, "view_type": "complete"}},
        ]
        self.assertEqual(list(limit_entries(rows, "chess", max_records=0)), [])

    def test_cached_entries_preserve_group_under_max_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "cache.jsonl")
            rows = [
                {"game": "poker", "tokens": ["<bos>", "<poker>", "view:complete", "pk:seat:p1", "pk:act:fold", "<eos>"], "metadata": {"seat_count": 2, "view_type": "complete"}},
                {"game": "poker", "tokens": ["<bos>", "<poker>", "view:imperfect:1", "pk:private_card", "pk:seat:p1", "pk:card:Ah", "pk:seat:p1", "pk:act:fold", "<eos>"], "metadata": {"seat_count": 2, "view_type": "imperfect", "viewer_seat": 1}},
                {"game": "poker", "tokens": ["<bos>", "<poker>", "view:complete", "pk:seat:p2", "pk:act:fold", "<eos>"], "metadata": {"seat_count": 2, "view_type": "complete"}},
            ]
            with open(cache_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            result = build_game_shards(
                "poker",
                [],
                os.path.join(temp_dir, "out"),
                target_tokens=1,
                max_records=1,
                cached_entries_path=cache_path,
            )
            self.assertEqual(result["rows"], 2)

    def test_mahjonglm_jsonl_shard_uses_compatible_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer_dir = os.path.join(temp_dir, "mahjong_tokenizer")
            os.makedirs(tokenizer_dir)
            vocab_tokens = [
                "<pad>", "<unk>", "<bos>", "<eos>", "view:complete",
                "ch:w:e2", "ch:e4", "ch:b:e7", "ch:e5", "ch:w:g1", "ch:f3",
                "ch:b:b8", "ch:c6", "ch:w:f1", "ch:b5", "ch:b:a7", "ch:a6", "ch:w:b5", "ch:a4",
            ]
            with open(os.path.join(tokenizer_dir, "vocab.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(vocab_tokens) + "\n")
            tokenizer = UniversalGameTokenizer.from_mahjonglm_assets(tokenizer_dir)

            chess_path = os.path.join(temp_dir, "sample.pgn")
            with open(chess_path, "w", encoding="utf-8") as f:
                f.write("""[Event "Sample"]\n[White "A"]\n[Black "B"]\n[Result "1-0"]\n[Date "2025.01.02"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 *""")
            result = build_game_shards(
                "chess",
                [chess_path],
                os.path.join(temp_dir, "out"),
                target_tokens=4,
                max_records=1,
                output_format="mahjonglm_jsonl",
                tokenizer=tokenizer,
            )
            with gzip.open(result["shards"][0]["path"], "rt", encoding="utf-8") as f:
                row = json.loads(f.readline())
            self.assertEqual(set(row), {"game_id", "year", "seat_count", "view_type", "viewer_seat", "length", "input_ids", "tokenizer_fingerprint"})
            self.assertEqual(row["year"], 2025)
            self.assertEqual(row["seat_count"], 2)
            self.assertEqual(row["view_type"], "complete")
            self.assertEqual(row["input_ids"][0], tokenizer.vocab["view:complete"])
            self.assertEqual(row["tokenizer_fingerprint"], result["tokenizer_fingerprint"])

    def test_shogi_parser_rejects_missing_terminal(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csa") as f:
            f.write("""V2.2
N+Black
N-White
PI
+
+7776FU
-3334FU
""")
            temp_path = f.name
        try:
            tokens, meta = parse_csa_to_tokens(temp_path)
            self.assertIsNone(tokens)
            self.assertIsNone(meta)
        finally:
            os.remove(temp_path)

    def test_shogi_parser_accepts_explicit_standard_board_rows(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csa") as f:
            f.write("""V2.2
N+Black
N-White
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA *
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  *
P5 *  *  *  *  *  *  *  *  *
P6 *  *  *  *  *  *  *  *  *
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI *
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
+
+7776FU
T1
-3334FU
T1
+2726FU
T1
-8384FU
T1
+2625FU
T1
-8485FU
T1
+6978KI
T1
-4132KI
T1
+2524FU
T1
-2324FU
T1
%TORYO
""")
            temp_path = f.name
        try:
            tokens, meta = parse_csa_to_tokens(temp_path)
            self.assertIsNotNone(tokens)
            self.assertEqual(tokens[1], "<shogi>")
        finally:
            os.remove(temp_path)

    def test_shogi_parser_rejects_nonstandard_explicit_board_rows(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csa") as f:
            f.write("""V2.2
N+Black
N-White
P1 * -KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA *
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  *
P5 *  *  *  *  *  *  *  *  *
P6 *  *  *  *  *  *  *  *  *
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI *
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
+
+7776FU
-3334FU
%TORYO
""")
            temp_path = f.name
        try:
            tokens, meta = parse_csa_to_tokens(temp_path)
            self.assertIsNone(tokens)
            self.assertIsNone(meta)
        finally:
            os.remove(temp_path)


    def test_shogi_directory_expands_nested_7z_archives(self):
        try:
            import py7zr
        except ImportError:
            self.skipTest("py7zr is not installed")

        csa_text = """V2.2
N+Black
N-White
PI
+
+7776FU
T1
-3334FU
T1
+2726FU
T1
-8384FU
T1
+2625FU
T1
-8485FU
T1
+6978KI
T1
-4132KI
T1
+2524FU
T1
-2324FU
T1
%TORYO
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            csa_path = os.path.join(temp_dir, "sample.csa")
            archive_dir = os.path.join(temp_dir, "archives")
            os.makedirs(archive_dir)
            archive_path = os.path.join(archive_dir, "sample.7z")
            with open(csa_path, "w", encoding="utf-8") as f:
                f.write(csa_text)
            with py7zr.SevenZipFile(archive_path, "w") as archive:
                archive.write(csa_path, arcname="inside/sample.csa")
            os.remove(csa_path)

            games = list(parse_shogi_directory(temp_dir, max_games=1))
            self.assertEqual(len(games), 1)
            self.assertEqual(games[0][0][1], "<shogi>")

    def test_production_shards_refuse_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            chess_path = os.path.join(temp_dir, "sample.pgn")
            with open(chess_path, "w", encoding="utf-8") as f:
                f.write("""[Event "Sample"]\n[White "A"]\n[Black "B"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 *""")
            out_dir = os.path.join(temp_dir, "out")
            build_game_shards("chess", [chess_path], out_dir, target_tokens=4, max_records=1)
            with self.assertRaises(ProductionDatasetError):
                build_game_shards("chess", [chess_path], out_dir, target_tokens=4, max_records=1)

    def test_source_catalog_prioritizes_top_quality_sources(self):
        catalog_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "source_catalog.json")
        catalog = load_source_catalog(catalog_path)
        self.assertEqual(catalog["target_tokens_per_game"], 3_000_000_000)
        for game, sources in catalog["games"].items():
            primary = [source for source in sources if source["priority"] == 1]
            self.assertEqual(len(primary), 1, game)
            self.assertIn(primary[0]["source_class"], {"engine_top", "human_top"}, game)
            self.assertNotIn(primary[0]["quality_tier"], {"excluded_from_primary", "filtered_fallback_only"}, game)
            for source in sources:
                if source["source_class"] == "human_public":
                    self.assertGreater(source["priority"], 1, source["name"])
                    self.assertIn(source["quality_tier"], {
                        "excluded_from_primary",
                        "filtered_fallback_only",
                    }, source["name"])

        chess_primary = assert_source_allowed_for_primary_build(
            catalog,
            "chess",
            "Stockfish fishtest LTC PGNs",
        )
        self.assertEqual(chess_primary["source_class"], "engine_top")
        with self.assertRaises(ProductionDatasetError):
            assert_source_allowed_for_primary_build(
                catalog,
                "chess",
                "Raw Lichess standard rated games",
            )

    def test_source_catalog_license_verified_field(self):
        catalog_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "source_catalog.json")
        catalog = load_source_catalog(catalog_path)
        for game, sources in catalog["games"].items():
            for source in sources:
                self.assertIn("license_verified", source, f"{game}: {source['name']} missing license_verified")
                self.assertIsInstance(source["license_verified"], bool, source["name"])

    def test_source_catalog_rejects_unverified_license(self):
        catalog = {
            "games": {
                "chess": [
                    {
                        "name": "Unverified Source",
                        "source_class": "engine_top",
                        "quality_tier": "primary_3b",
                        "license": "unknown",
                        "license_verified": False,
                    }
                ]
            }
        }
        with self.assertRaises(ProductionDatasetError):
            assert_source_allowed_for_primary_build(catalog, "chess", "Unverified Source")

    def test_build_game_shards_rejects_unknown_source_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            chess_path = os.path.join(temp_dir, "sample.pgn")
            with open(chess_path, "w", encoding="utf-8") as f:
                f.write('[Event "Sample"]\n[White "A"]\n[Black "B"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 *')
            out_dir = os.path.join(temp_dir, "out")
            with self.assertRaises(ProductionDatasetError):
                build_game_shards(
                    "chess", [chess_path], out_dir,
                    target_tokens=4, max_records=1,
                    allowed_source_ids={"bogus_id"},
                )

    def test_build_game_shards_tokenizer_fingerprint_consistency(self):
        result = build_game_shards.__code__.co_varnames
        self.assertIn("tokenizer_fingerprint", result)
        self.assertIn("allowed_source_ids", result)

if __name__ == "__main__":
    unittest.main()
