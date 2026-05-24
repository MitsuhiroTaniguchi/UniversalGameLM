import os
import json

class UniversalGameTokenizer:
    """
    Universal Tokenizer that manages a shared vocabulary across different games.
    """
    def __init__(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<chess>", "<shogi>"]
            
        self.special_tokens = special_tokens
        self.vocab = {}
        self.inv_vocab = {}
        
        # Initialize with special tokens
        for token in self.special_tokens:
            self._register_token(token)

    def _register_token(self, token):
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inv_vocab[idx] = token
            return idx
        return self.vocab[token]

    def build_vocab(self, token_generator):
        """
        Iterates over a generator of game token lists and registers all unique moves.
        """
        print("[Tokenizer] Building vocabulary from dataset...")
        move_count = 0
        for tokens in token_generator:
            for token in tokens:
                if token not in self.vocab:
                    self._register_token(token)
                    move_count += 1
        print(f"[Tokenizer] Vocabulary built. Total size: {len(self.vocab)} (Added {move_count} moves).")

    def encode(self, tokens):
        """Converts list of string tokens to a list of integer IDs."""
        unk_id = self.vocab.get("<unk>", 1)
        return [self.vocab.get(token, unk_id) for token in tokens]

    def decode(self, ids):
        """Converts list of integer IDs back to string tokens."""
        unk_token = "<unk>"
        return [self.inv_vocab.get(idx, unk_token) for idx in ids]

    def save_vocab(self, filepath):
        """Saves vocabulary mapping to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
        print(f"[Tokenizer] Saved vocabulary to {filepath}")

    def load_vocab(self, filepath):
        """Loads vocabulary mapping from a JSON file."""
        if not os.path.exists(filepath):
            print(f"[Error] Vocabulary file not found: {filepath}")
            return False
            
        with open(filepath, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
            
        self.inv_vocab = {int(v): k for k, v in self.vocab.items()}
        # Re-derive special tokens list based on loaded keys
        self.special_tokens = [k for k in self.vocab.keys() if k.startswith("<") and k.endswith(">")]
        print(f"[Tokenizer] Loaded vocabulary from {filepath}. Size: {len(self.vocab)}")
        return True

    @property
    def vocab_size(self):
        return len(self.vocab)

if __name__ == "__main__":
    # Test tokenizer
    tokenizer = UniversalGameTokenizer()
    print("Initial vocab:", tokenizer.vocab)
    
    # Register some mock games
    mock_games = [
        ["<bos>", "<chess>", "e2e4", "e7e5", "g1f3", "<eos>"],
        ["<bos>", "<shogi>", "7g7f", "1c1d", "2g2f", "<eos>"]
    ]
    
    tokenizer.build_vocab(mock_games)
    
    # Test encode/decode
    test_game = ["<bos>", "<chess>", "e2e4", "e7e5", "invalid_move", "<eos>"]
    encoded = tokenizer.encode(test_game)
    decoded = tokenizer.decode(encoded)
    
    print("\nTest Game:", test_game)
    print("Encoded IDs:", encoded)
    print("Decoded Tokens:", decoded)
    
    # Test save/load
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_path = os.path.join(base_dir, "tokenized_data", "vocab.json")
    tokenizer.save_vocab(vocab_path)
    
    new_tokenizer = UniversalGameTokenizer()
    new_tokenizer.load_vocab(vocab_path)
    print("Loaded vocab size matches:", new_tokenizer.vocab_size == tokenizer.vocab_size)
    
    # Clean up test vocab
    if os.path.exists(vocab_path):
        os.remove(vocab_path)
