import os
import json
import hashlib
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import AddedToken
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

class UniversalGameTokenizer:
    """
    Universal Tokenizer that manages a shared vocabulary across different games.
    """
    def __init__(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = [
                "<pad>", "<unk>", "<bos>", "<eos>",
                "<mahjong>", "<chess>", "<shogi>", "<go>", "<othello>", "<poker>", "<bridge>",
            ]
            
        self.special_tokens = special_tokens
        self.vocab = {}
        self.inv_vocab = {}
        self.backend_tokenizer = None
        
        # Initialize with special tokens
        for token in self.special_tokens:
            self._register_token(token)

    def _register_token(self, token):
        if token not in self.vocab:
            idx = (max(self.inv_vocab) + 1) if self.inv_vocab else 0
            self.vocab[token] = idx
            self.inv_vocab[idx] = token
            if self.backend_tokenizer is not None:
                self.backend_tokenizer.add_tokens([AddedToken(token, single_word=False, lstrip=False, rstrip=False, normalized=False)])
                backend_id = self.backend_tokenizer.token_to_id(token)
                if backend_id != idx:
                    raise ValueError(f"Backend tokenizer id mismatch for {token}: expected {idx}, got {backend_id}")
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

    def encode_strict(self, tokens):
        """Converts tokens to ids and fails if any token is absent."""
        missing = [token for token in tokens if token not in self.vocab]
        if missing:
            raise KeyError(f"Tokenizer is missing tokens: {missing[:10]}")
        return [self.vocab[token] for token in tokens]

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
        self._validate_vocab_ids()
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
            instance.backend_tokenizer = tokenizer
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
        instance._validate_vocab_ids()
        instance.special_tokens = [
            token for token in instance.vocab
            if token.startswith("<") and token.endswith(">")
        ]
        return instance

    @classmethod
    def from_universal_vocab(cls, vocab_path=None):
        """
        Loads the unified universal vocabulary (vocab/universal.txt or .json).

        All game tokens are pre-registered. Game marker special tokens
        (``<chess>``, ``<shogi>``, etc.) are NOT in the universal vocab
        file — use :meth:`add_tokens` or :meth:`_register_token` to add
        them before encoding.
        """
        import json as _json
        from pathlib import Path as _Path

        if vocab_path is None:
            vocab_path = _Path(__file__).parent.parent / "vocab" / "universal.json"
        else:
            vocab_path = _Path(vocab_path)

        instance = cls(special_tokens=[])

        if vocab_path.suffix == ".json":
            with open(vocab_path, "r", encoding="utf-8") as f:
                raw = _json.load(f)
            instance.vocab = {token: int(idx) for token, idx in raw.items()}
        elif vocab_path.suffix == ".txt":
            with open(vocab_path, "r", encoding="utf-8") as f:
                instance.vocab = {line.rstrip("\n"): idx for idx, line in enumerate(f)}
        else:
            raise ValueError(f"Unsupported vocab format: {vocab_path}")

        instance.inv_vocab = {int(idx): token for token, idx in instance.vocab.items()}
        instance._validate_vocab_ids()
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

    def _validate_vocab_ids(self):
        ids = list(self.vocab.values())
        if len(ids) != len(set(ids)):
            raise ValueError("Tokenizer vocabulary contains duplicate ids")
        if any(not isinstance(idx, int) or idx < 0 for idx in ids):
            raise ValueError("Tokenizer vocabulary ids must be non-negative integers")
        expected = set(range(max(ids) + 1)) if ids else set()
        missing = sorted(expected - set(ids))
        if missing:
            raise ValueError(f"Tokenizer vocabulary ids must be dense; missing ids start with {missing[:10]}")

    def fingerprint(self):
        payload = json.dumps(
            sorted(self.vocab.items(), key=lambda item: item[1]),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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

        if self.backend_tokenizer is not None:
            tokenizer = self.backend_tokenizer
        else:
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
