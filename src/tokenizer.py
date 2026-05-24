import os
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

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

    @classmethod
    def from_mahjonglm_assets(cls, tokenizer_dir):
        """
        Loads MahjongLM tokenizer assets and preserves every existing token id.

        Supported inputs:
        - tokenizer.json
        - vocab.txt
        - vocab.json
        """
        tokenizer_dir = Path(tokenizer_dir)
        instance = cls(special_tokens=[])
        tokenizer_json = tokenizer_dir / "tokenizer.json"
        vocab_txt = tokenizer_dir / "vocab.txt"
        vocab_json = tokenizer_dir / "vocab.json"

        if tokenizer_json.exists():
            tokenizer = Tokenizer.from_file(str(tokenizer_json))
            vocab = tokenizer.get_vocab()
            instance.vocab = {token: int(idx) for token, idx in vocab.items()}
        elif vocab_txt.exists():
            with open(vocab_txt, "r", encoding="utf-8") as f:
                instance.vocab = {line.rstrip("\n"): idx for idx, line in enumerate(f)}
        elif vocab_json.exists():
            with open(vocab_json, "r", encoding="utf-8") as f:
                instance.vocab = {token: int(idx) for token, idx in json.load(f).items()}
        else:
            raise FileNotFoundError(f"No tokenizer.json, vocab.txt, or vocab.json found in {tokenizer_dir}")

        instance.inv_vocab = {idx: token for token, idx in instance.vocab.items()}
        instance.special_tokens = [
            token for token in instance.vocab
            if token.startswith("<") and token.endswith(">")
        ]
        return instance

    def add_tokens(self, tokens):
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self._register_token(token)
                added += 1
        return added

    def save_mahjonglm_assets(self, output_dir):
        """Writes tokenizer assets with MahjongLM ids preserved and new ids appended."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ordered = [token for token, _ in sorted(self.vocab.items(), key=lambda item: item[1])]

        with open(output_dir / "vocab.txt", "w", encoding="utf-8") as f:
            for token in ordered:
                f.write(token + "\n")
        with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)

        tokenizer = Tokenizer(WordLevel(vocab=self.vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.save(str(output_dir / "tokenizer.json"))
        with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump({"tokenizer_class": "PreTrainedTokenizerFast"}, f, indent=2)
        with open(output_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
            json.dump({
                "unk_token": "<unk>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
            }, f, indent=2)

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
