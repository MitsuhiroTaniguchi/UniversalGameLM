import unittest
import os
import tempfile
from src.chess_parser import parse_pgn_to_tokens
from src.shogi_parser import parse_csa_to_tokens
from src.go_parser import parse_sgf_to_tokens
from src.othello_parser import parse_othello_pgn_to_tokens
from src.poker_parser import PokerHandSimulator
from src.tokenizer import UniversalGameTokenizer

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
            self.assertEqual(tokens[2], "pd")
            self.assertEqual(tokens[3], "dd")
            self.assertEqual(tokens[4], "pp")
            self.assertEqual(tokens[-1], "<eos>")
            self.assertEqual(meta["black"], "BlackPlayer")
            self.assertEqual(meta["white"], "WhitePlayer")
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
        self.assertTrue(any(t.startswith("H:") for t in tokens))
        self.assertTrue(any(t.startswith("WINNER:") for t in tokens))
        self.assertIsNotNone(meta["winner"])

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

if __name__ == "__main__":
    unittest.main()
