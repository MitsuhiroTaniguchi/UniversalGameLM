import os
import gzip
from pathlib import Path
import sgfmill.sgf
import sgfmill.sgf_grammar
import sgfmill.boards

def get_sgf_property(node, name, default="Unknown"):
    """Helper to safely extract SGF properties without raising KeyError."""
    try:
        value = node.get(name)
    except KeyError:
        return default
    return default if value is None else value

def coords_to_token(coords, board_size):
    """Converts sgfmill row/col coordinates back to SGF-style two-letter tokens."""
    row, col = coords
    col_letter = chr(ord('a') + col)
    row_letter = chr(ord('a') + (board_size - 1 - row))
    return f"{col_letter}{row_letter}"


def token_to_coords(token, board_size):
    col = ord(token[0]) - ord('a')
    row = board_size - 1 - (ord(token[1]) - ord('a'))
    return row, col


def _board_position_key(board):
    return tuple(tuple(board.get(row, col) for col in range(board.side)) for row in range(board.side))


def validate_go_token_sequence(tokens):
    if len(tokens) < 4 or tokens[0] != "<bos>" or tokens[1] != "<go>" or tokens[-1] != "<eos>":
        raise ValueError("Invalid Go BOS/game/EOS markers")
    if not tokens[2].startswith("SZ:"):
        raise ValueError("Missing Go board size token")
    board_size = int(tokens[2].split(":", 1)[1])
    board = sgfmill.boards.Board(board_size)
    position_history = [_board_position_key(board)]
    rule_text = " ".join(token.split(":", 1)[1].lower() for token in tokens[3:-1] if token.startswith("RU:"))
    use_positional_superko = any(name in rule_text for name in ("aga", "chinese", "nz", "new_zealand", "tromp"))
    seen_setup = True
    for token in tokens[3:-1]:
        if token.startswith(("KM:", "RU:", "HA:")):
            continue
        if token.startswith(("AB:", "AW:", "AE:")):
            if not seen_setup:
                raise ValueError("Setup tokens after moves are not supported")
            point = token.split(":", 1)[1]
            coords = token_to_coords(point, board_size)
            if token.startswith("AB:"):
                board.apply_setup([coords], [], [])
            elif token.startswith("AW:"):
                board.apply_setup([], [coords], [])
            else:
                board.apply_setup([], [], [coords])
            position_history = [_board_position_key(board)]
            continue
        seen_setup = False
        if token in {"b:pass", "w:pass"}:
            continue
        if token.startswith(("b:", "w:")):
            color, point = token.split(":", 1)
            row, col = token_to_coords(point, board_size)
            board.play(row, col, color)
            position_key = _board_position_key(board)
            if use_positional_superko and position_key in position_history:
                raise ValueError("Go sequence violates positional superko")
            if not use_positional_superko and len(position_history) >= 2 and position_key == position_history[-2]:
                raise ValueError("Go sequence violates simple ko")
            position_history.append(position_key)
            continue
        raise ValueError(f"Invalid Go token: {token}")
    return True


def _coarse_tree_has_variations(tree):
    children = getattr(tree, "children", [])
    if len(children) > 1:
        return True
    return any(_coarse_tree_has_variations(child) for child in children)

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

        coarse_tree = sgfmill.sgf_grammar.parse_sgf_game(sgf_content.encode("utf-8", errors="ignore"))
        if _coarse_tree_has_variations(coarse_tree):
            raise ValueError("SGF variations are not accepted in production tokenization")

        game = sgfmill.sgf.Sgf_game.from_string(sgf_content)
        root = game.get_root()

        board_size = root.get_size()
        if board_size < 1 or board_size > 26:
            raise ValueError(f"Unsupported Go board size for tokenization: {board_size}")

        # Extract setup stones before moves. Without these, handicap/setup SGFs
        # become ambiguous or illegal when reconstructed from tokens.
        setup_tokens = []
        def append_setup_tokens(node, allow_non_root=False):
            if not node.has_setup_stones():
                return
            if not allow_non_root and node is not root:
                raise ValueError("Non-root setup nodes are not accepted")
            black_stones, white_stones, empty_points = node.get_setup_stones()
            for coords in list(black_stones) + list(white_stones) + list(empty_points):
                row, col = coords
                if not (0 <= row < board_size and 0 <= col < board_size):
                    raise ValueError(f"Setup point out of range: {coords}")
            setup_tokens.extend(f"AB:{coords_to_token(c, board_size)}" for c in sorted(black_stones))
            setup_tokens.extend(f"AW:{coords_to_token(c, board_size)}" for c in sorted(white_stones))
            setup_tokens.extend(f"AE:{coords_to_token(c, board_size)}" for c in sorted(empty_points))

        append_setup_tokens(root, allow_non_root=True)

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

        context_tokens = [f"SZ:{board_size}"]
        for prefix, prop in (("KM", "KM"), ("RU", "RU"), ("HA", "HA")):
            value = get_sgf_property(root, prop, None)
            if value not in (None, "Unknown", ""):
                context_tokens.append(f"{prefix}:{str(value).replace(' ', '_')}")
        tokens = ["<bos>", "<go>"] + context_tokens + setup_tokens + moves + ["<eos>"]
        validate_go_token_sequence(tokens)

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
            "seat_count": 2,
            "view_type": "complete",
            "viewer_seat": None,
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
    for root, dirs, files in os.walk(path):
        dirs.sort()
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
