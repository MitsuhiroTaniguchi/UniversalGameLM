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
    if not tokens[2].startswith("go:sz:"):
        raise ValueError("Missing Go board size token")
    board_size = int(tokens[2].split(":", 2)[2])
    board = sgfmill.boards.Board(board_size)
    position_history = [_board_position_key(board)]
    rule_text = " ".join(token.split(":", 2)[2].lower() for token in tokens[3:-1] if token.startswith("go:ru:"))
    use_positional_superko = any(name in rule_text for name in ("aga", "chinese", "nz", "new_zealand", "tromp"))
    seen_setup = True
    body = tokens[3:-1]
    i = 0
    while i < len(body):
        token = body[i]
        if token.startswith(("go:km:", "go:ru:", "go:ha:", "go:result:", "go:end:", "go:score:", "go:num:", "go:move:")):
            i += 1
            continue
        if token in ("go:setup_b", "go:setup_w", "go:setup_e"):
            if not seen_setup:
                raise ValueError("Setup tokens after moves are not supported")
            if i + 1 >= len(body):
                raise ValueError("Setup action without coordinate")
            point = body[i + 1].split(":", 1)[1]
            coords = token_to_coords(point, board_size)
            if token == "go:setup_b":
                board.apply_setup([coords], [], [])
            elif token == "go:setup_w":
                board.apply_setup([], [coords], [])
            else:
                board.apply_setup([], [], [coords])
            position_history = [_board_position_key(board)]
            i += 2
            continue
        if token.startswith(("go:b:", "go:w:")):
            seen_setup = False
            parts = token.split(":", 2)
            color = parts[1]
            point = parts[2]
            if point == "pass":
                i += 1
                continue
            row, col = token_to_coords(point, board_size)
            board.play(row, col, color)
            position_key = _board_position_key(board)
            if use_positional_superko and position_key in position_history:
                raise ValueError("Go sequence violates positional superko")
            if not use_positional_superko and len(position_history) >= 2 and position_key == position_history[-2]:
                raise ValueError("Go sequence violates simple ko")
            position_history.append(position_key)
            i += 1
            continue
        raise ValueError(f"Invalid Go token: {token}")
    return True


def _coarse_tree_has_variations(tree):
    children = getattr(tree, "children", [])
    if len(children) > 1:
        return True
    return any(_coarse_tree_has_variations(child) for child in children)


def _result_tokens(result):
    if result in (None, "Unknown", "", "*"):
        return ["go:result:unknown", "go:end:unknown"]
    value = str(result).strip()
    lower = value.lower().replace(" ", "_")
    winner = "black" if lower.startswith("b+") else "white" if lower.startswith("w+") else "unknown"
    end = "unknown"
    score = None
    if "+" in lower:
        suffix = lower.split("+", 1)[1]
        if suffix in {"r", "resign"}:
            end = "resign"
        elif suffix in {"t", "time"}:
            end = "time"
        elif suffix in {"f", "forfeit"}:
            end = "forfeit"
        else:
            end = "points"
            score = suffix
    tokens = [f"go:result:{winner}_win", f"go:end:{end}"]
    if score:
        tokens.extend(_numeric_value_tokens("go:score", "go:num", score))
    return tokens


def _numeric_value_tokens(field_prefix, digit_prefix, value):
    pieces = []
    for char in str(value).strip():
        if char.isdigit():
            pieces.append(f"{digit_prefix}:{char}")
        elif char == ".":
            pieces.append(f"{digit_prefix}:dot")
        elif char in "+-":
            pieces.append(f"{digit_prefix}:{'plus' if char == '+' else 'neg'}")
        else:
            return []
    return [f"{field_prefix}:BEGIN"] + pieces + [f"{field_prefix}:END"] if pieces else []

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
            for c in sorted(black_stones):
                setup_tokens.extend(["go:setup_b", f"go:{coords_to_token(c, board_size)}"])
            for c in sorted(white_stones):
                setup_tokens.extend(["go:setup_w", f"go:{coords_to_token(c, board_size)}"])
            for c in sorted(empty_points):
                setup_tokens.extend(["go:setup_e", f"go:{coords_to_token(c, board_size)}"])

        append_setup_tokens(root, allow_non_root=True)

        moves = []
        for node in game.get_main_sequence():
            if node is not root:
                append_setup_tokens(node)
            move = node.get_move()
            if move is not None and move[0] is not None:
                color, coords = move
                if coords is None:
                    moves.append(f"go:{color}:pass")
                else:
                    row, col = coords
                    if not (0 <= row < board_size and 0 <= col < board_size):
                        raise ValueError(f"Move point out of range: {coords}")
                    moves.append(f"go:{color}:{coords_to_token(coords, board_size)}")

        # Quality Filter: Skip empty or extremely short games
        if len(moves) < 10:
            return None, None

        context_tokens = [f"go:sz:{board_size}"]
        for prefix, prop in (("km", "KM"), ("ru", "RU"), ("ha", "HA")):
            value = get_sgf_property(root, prop, None)
            if value not in (None, "Unknown", ""):
                context_tokens.append(f"go:{prefix}:{str(value).lower().replace(' ', '_')}")
        result = get_sgf_property(root, "RE", "*")
        tokens = ["<bos>", "<go>"] + context_tokens + setup_tokens + moves + _result_tokens(result) + ["<eos>"]
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
            "setup_count": len(setup_tokens) // 2,
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
