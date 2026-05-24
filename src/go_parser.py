import os
import glob
import sgfmill.sgf

def get_sgf_property(node, name, default="Unknown"):
    """Helper to safely extract SGF properties without raising KeyError."""
    try:
        return node.get(name) or default
    except KeyError:
        return default

def parse_sgf_to_tokens(sgf_path):
    """
    Parses a Go SGF file using sgfmill and returns the token sequence and metadata.
    Token sequence format: ['<bos>', '<go>', 'pd', 'dd', 'pp', 'pass', ..., '<eos>']
    """
    if not os.path.exists(sgf_path):
        print(f"[Error] SGF file not found: {sgf_path}")
        return None, None

    try:
        with open(sgf_path, "r", encoding="utf-8", errors="ignore") as f:
            sgf_content = f.read()

        game = sgfmill.sgf.Sgf_game.from_string(sgf_content)
        root = game.get_root()

        # Extract moves
        usi_moves = []
        for node in game.get_main_sequence():
            move = node.get_move()
            if move is not None and move[0] is not None:
                color, coords = move
                if coords is None:
                    usi_moves.append("pass")
                else:
                    row, col = coords
                    # Map bottom-left coords to top-left letter coordinates (0 -> a, 1 -> b, etc.)
                    col_letter = chr(ord('a') + col)
                    row_letter = chr(ord('a') + (18 - row))
                    usi_moves.append(f"{col_letter}{row_letter}")

        # Quality Filter: Skip empty or extremely short games
        if len(usi_moves) < 10:
            return None, None

        tokens = ["<bos>", "<go>"] + usi_moves + ["<eos>"]

        metadata = {
            "black": get_sgf_property(root, "PB", "Unknown"),
            "white": get_sgf_property(root, "PW", "Unknown"),
            "result": get_sgf_property(root, "RE", "*"),
            "date": get_sgf_property(root, "DT", "????-??-??"),
            "move_count": len(usi_moves),
            "filename": os.path.basename(sgf_path)
        }

        return tokens, metadata
    except Exception as e:
        print(f"[Warning] Failed to parse {os.path.basename(sgf_path)}: {e}")
        return None, None

def parse_go_directory(directory_path, max_games=None):
    """
    Parses all SGF files in a directory and yields token sequences.
    """
    sgf_pattern = os.path.join(directory_path, "**", "*.sgf")
    sgf_files = glob.glob(sgf_pattern, recursive=True)
    
    print(f"[Parsing Go] Found {len(sgf_files)} SGF files in {directory_path}...")
    
    games_parsed = 0
    for filepath in sgf_files:
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
