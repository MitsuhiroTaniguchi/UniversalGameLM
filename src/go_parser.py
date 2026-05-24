import os
import gzip
from pathlib import Path
import sgfmill.sgf

def get_sgf_property(node, name, default="Unknown"):
    """Helper to safely extract SGF properties without raising KeyError."""
    try:
        return node.get(name) or default
    except KeyError:
        return default

def coords_to_token(coords, board_size):
    """Converts sgfmill row/col coordinates back to SGF-style two-letter tokens."""
    row, col = coords
    col_letter = chr(ord('a') + col)
    row_letter = chr(ord('a') + (board_size - 1 - row))
    return f"{col_letter}{row_letter}"

def parse_sgf_to_tokens(sgf_path):
    """
    Parses a Go SGF file using sgfmill and returns the token sequence and metadata.
    Token sequence format: ['<bos>', '<go>', 'pd', 'dd', 'pp', 'pass', ..., '<eos>']
    """
    if not os.path.exists(sgf_path):
        print(f"[Error] SGF file not found: {sgf_path}")
        return None, None

    try:
        opener = gzip.open if str(sgf_path).lower().endswith(".gz") else open
        with opener(sgf_path, "rt", encoding="utf-8", errors="ignore") as f:
            sgf_content = f.read()

        game = sgfmill.sgf.Sgf_game.from_string(sgf_content)
        root = game.get_root()
        if any(len(getattr(node, "_children", [])) > 1 for node in game.get_main_sequence()):
            raise ValueError("SGF variations are not accepted in production tokenization")

        board_size = root.get_size()
        if board_size < 1 or board_size > 26:
            raise ValueError(f"Unsupported Go board size for tokenization: {board_size}")

        # Extract setup stones before moves. Without these, handicap/setup SGFs
        # become ambiguous or illegal when reconstructed from tokens.
        setup_tokens = []
        def append_setup_tokens(node):
            if not node.has_setup_stones():
                return
            black_stones, white_stones, empty_points = node.get_setup_stones()
            for coords in list(black_stones) + list(white_stones) + list(empty_points):
                row, col = coords
                if not (0 <= row < board_size and 0 <= col < board_size):
                    raise ValueError(f"Setup point out of range: {coords}")
            setup_tokens.extend(f"AB:{coords_to_token(c, board_size)}" for c in sorted(black_stones))
            setup_tokens.extend(f"AW:{coords_to_token(c, board_size)}" for c in sorted(white_stones))
            setup_tokens.extend(f"AE:{coords_to_token(c, board_size)}" for c in sorted(empty_points))

        append_setup_tokens(root)

        moves = []
        for node in game.get_main_sequence():
            if node is not root:
                append_setup_tokens(node)
            move = node.get_move()
            if move is not None and move[0] is not None:
                color, coords = move
                if coords is None:
                    moves.append(f"{color}:pass")
                else:
                    row, col = coords
                    if not (0 <= row < board_size and 0 <= col < board_size):
                        raise ValueError(f"Move point out of range: {coords}")
                    moves.append(f"{color}:{coords_to_token(coords, board_size)}")

        # Quality Filter: Skip empty or extremely short games
        if len(moves) < 10:
            return None, None

        tokens = ["<bos>", "<go>", f"SZ:{board_size}"] + setup_tokens + moves + ["<eos>"]

        metadata = {
            "black": get_sgf_property(root, "PB", "Unknown"),
            "white": get_sgf_property(root, "PW", "Unknown"),
            "result": get_sgf_property(root, "RE", "*"),
            "date": get_sgf_property(root, "DT", "????-??-??"),
            "board_size": board_size,
            "komi": get_sgf_property(root, "KM", None),
            "handicap": get_sgf_property(root, "HA", None),
            "rules": get_sgf_property(root, "RU", None),
            "setup_count": len(setup_tokens),
            "move_count": len(moves),
            "filename": os.path.basename(sgf_path),
            "source_path": str(Path(sgf_path).resolve()),
        }

        return tokens, metadata
    except Exception as e:
        print(f"[Warning] Failed to parse {os.path.basename(sgf_path)}: {e}")
        return None, None

def iter_sgf_files(directory_path):
    path = Path(directory_path)
    if path.is_file():
        if path.name.lower().endswith((".sgf", ".sgf.gz")):
            yield str(path)
        return
    for root, _, files in os.walk(path):
        for name in sorted(files):
            if name.lower().endswith((".sgf", ".sgf.gz")):
                yield str(Path(root) / name)


def parse_go_directory(directory_path, max_games=None):
    """
    Parses all SGF files in a directory and yields token sequences.
    """
    print(f"[Parsing Go] Streaming SGF files from {directory_path}...")
    
    games_parsed = 0
    for filepath in iter_sgf_files(directory_path):
        tokens, metadata = parse_sgf_to_tokens(filepath)
        if tokens is not None:
            games_parsed += 1
            yield tokens, metadata
            
            if max_games and games_parsed >= max_games:
                break
                
    print(f"[Success] Parsed {games_parsed} Go games.")

if __name__ == "__main__":
    # Test parser
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_go_dir = os.path.join(base_dir, "data", "go")
    
    if os.path.exists(test_go_dir):
        for i, (tokens, meta) in enumerate(parse_go_directory(test_go_dir, max_games=3)):
            print(f"\nGame #{i+1} Metadata: {meta}")
            print(f"Tokens (length {len(tokens)}): {tokens[:15]} ... {tokens[-5:]}")
    else:
        print("[Info] Please run src/download.py first to download the test Go logs.")
