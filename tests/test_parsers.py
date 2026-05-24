import unittest
import os
import tempfile
import zipfile
import gzip
import json
from src.chess_parser import parse_pgn_to_tokens
from src.shogi_parser import parse_csa_to_tokens
from src.go_parser import parse_sgf_to_tokens
from src.othello_parser import parse_othello_pgn_to_tokens
from src.poker_parser import PokerHandSimulator
from src.poker_parser import parse_phh_to_tokens
from src.tokenizer import UniversalGameTokenizer
from src.download import safe_extract_zip
from src.hf_uploader import HuggingFaceShardUploader
from src.production_pipeline import build_game_shards, validate_entry, ProductionDatasetError

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
            self.assertEqual(tokens[2], "e2e4")
            self.assertEqual(tokens[-1], "<eos>")
            self.assertEqual(meta["white"], "Player1")
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
            self.assertEqual(tokens[2], "SZ:19")
            self.assertEqual(tokens[3], "b:pd")
            self.assertEqual(tokens[4], "w:dd")
            self.assertEqual(tokens[5], "b:pp")
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
            self.assertIn("AB:pd", tokens)
            self.assertIn("AB:dd", tokens)
            self.assertIn("AW:pp", tokens)
            self.assertEqual(meta["setup_count"], 3)
        finally:
            os.remove(temp_path)

    def test_othello_parser(self):
        # Create a mock Othello PGN with 8 moves to pass the quality filter
        mock_pgn = """[Event "Othello Match"]
[Black "PlayerB"]
[White "PlayerW"]
[Result "32-32"]

1. F5 D6 2. C3 F3 3. F4 D3 4. C4 G6 *"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(mock_pgn)
            temp_path = f.name
        try:
            games = list(parse_othello_pgn_to_tokens(temp_path))
            self.assertEqual(len(games), 1)
            tokens, meta = games[0]
            self.assertEqual(tokens[0], "<bos>")
            self.assertEqual(tokens[1], "<othello>")
            self.assertEqual(tokens[2], "f5")
            self.assertEqual(tokens[3], "d6")
            self.assertEqual(tokens[4], "c3")
            self.assertEqual(tokens[5], "f3")
            self.assertEqual(tokens[-1], "<eos>")
            self.assertEqual(meta["black"], "PlayerB")
        finally:
            os.remove(temp_path)

    def test_poker_simulator(self):
        simulator = PokerHandSimulator()
        tokens, meta = simulator.simulate_hand()
        self.assertEqual(tokens[0], "<bos>")
        self.assertEqual(tokens[1], "<poker>")
        self.assertEqual(tokens[-1], "<eos>")
        self.assertFalse(any(t.startswith("H:") for t in tokens))
        self.assertTrue(any(t.startswith("WINNER:") for t in tokens))
        self.assertIsNotNone(meta["winner"])
        self.assertEqual(meta["source"], "synthetic_simulator")

    def test_poker_score_compares_tie_breakers(self):
        simulator = PokerHandSimulator()
        board = ["2c", "3s", "4h", "8d", "9c"]
        kings = simulator._score_key(simulator.get_best_hand(["Kh", "Kd"], board))
        aces = simulator._score_key(simulator.get_best_hand(["Ah", "Ad"], board))
        self.assertGreater(aces, kings)

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
                f.write("""[Event "Othello"]\n[Black "B"]\n[White "W"]\n[Result "32-32"]\n\n1. F5 D6 2. C3 F3 3. F4 D3 4. C4 G6 *""")

            poker_path = os.path.join(temp_dir, "sample.phh")
            with open(poker_path, "w", encoding="utf-8") as f:
                f.write('''actions = ["deal_hole P1 AhAd", "post_blind P1 50", "post_blind P2 100", "call P1", "check P2", "deal_board 2c3d4h", "bet P2 200", "fold P1"]\n''')

            cases = {
                "chess": [chess_path],
                "shogi": [shogi_dir],
                "go": [go_dir],
                "othello": [othello_path],
                "poker": [poker_path],
            }
            for game, paths in cases.items():
                out_dir = os.path.join(temp_dir, "out", game)
                result = build_game_shards(
                    game,
                    paths,
                    out_dir,
                    target_tokens=4,
                    max_tokens_per_shard=100,
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
            "show_or_muck_hole_cards_p1_ahad",
        ]
        for token in private_tokens:
            with self.subTest(token=token):
                with self.assertRaises(ProductionDatasetError):
                    validate_entry({"game": "poker", "tokens": ["<bos>", "<poker>", token, "<eos>"]})

    def test_phh_parser_only_reads_actions_field(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".phh") as f:
            f.write('''venue = "quoted venue must not become action"\nplayers = ["Alice", "Bob"]\nactions = [\n  "deal_hole P1 AhAd",\n  "post_blind P1 50",\n  "post_blind P2 100",\n  "call P1"\n]\n''')
            temp_path = f.name
        try:
            hands = list(parse_phh_to_tokens(temp_path))
            self.assertEqual(len(hands), 1)
            tokens, meta = hands[0]
            self.assertNotIn("quoted_venue_must_not_become_action", tokens)
            self.assertNotIn("alice", tokens)
            self.assertNotIn("deal_hole_p1_ahad", tokens)
            self.assertEqual(tokens[2:], ["post_blind_p1_50", "post_blind_p2_100", "call_p1", "<eos>"])
            self.assertEqual(meta["private_actions_excluded"], 1)
        finally:
            os.remove(temp_path)

    def test_production_shards_refuse_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            chess_path = os.path.join(temp_dir, "sample.pgn")
            with open(chess_path, "w", encoding="utf-8") as f:
                f.write("""[Event "Sample"]\n[White "A"]\n[Black "B"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 *""")
            out_dir = os.path.join(temp_dir, "out")
            build_game_shards("chess", [chess_path], out_dir, target_tokens=4, max_records=1)
            with self.assertRaises(ProductionDatasetError):
                build_game_shards("chess", [chess_path], out_dir, target_tokens=4, max_records=1)

if __name__ == "__main__":
    unittest.main()
