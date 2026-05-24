import json
import os
import re
import gzip
from pathlib import Path


BOARD_SIZE = 8
PASS_TOKEN = "pass"
DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _initial_board():
    board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    board[3][3] = "W"
    board[3][4] = "B"
    board[4][3] = "B"
    board[4][4] = "W"
    return board


def _coord_to_xy(move):
    if not re.fullmatch(r"[a-h][1-8]", move):
        raise ValueError(f"Invalid Othello coordinate: {move}")
    return ord(move[0]) - ord("a"), int(move[1]) - 1


def _captures(board, color, x, y):
    if board[y][x] is not None:
        return []
    opponent = "W" if color == "B" else "B"
    captured = []
    for dx, dy in DIRECTIONS:
        line = []
        cx, cy = x + dx, y + dy
        while 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE and board[cy][cx] == opponent:
            line.append((cx, cy))
            cx += dx
            cy += dy
        if line and 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE and board[cy][cx] == color:
            captured.extend(line)
    return captured


def legal_moves(board, color):
    moves = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if _captures(board, color, x, y):
                moves.append(f"{chr(ord('a') + x)}{y + 1}")
    return moves


def apply_move(board, color, move):
    if move == PASS_TOKEN:
        if legal_moves(board, color):
            raise ValueError(f"Illegal pass while {color} has legal moves")
        return
    x, y = _coord_to_xy(move)
    captured = _captures(board, color, x, y)
    if not captured:
        raise ValueError(f"Illegal Othello move: {move}")
    board[y][x] = color
    for cx, cy in captured:
        board[cy][cx] = color


def validate_othello_moves(moves):
    board = _initial_board()
    color = "B"
    consecutive_passes = 0
    for index, move in enumerate(moves):
        canonical = PASS_TOKEN if move in ("pa", "pass") else move
        apply_move(board, color, canonical)
        consecutive_passes = consecutive_passes + 1 if canonical == PASS_TOKEN else 0
        color = "W" if color == "B" else "B"
        if consecutive_passes >= 2:
            remaining = moves[index + 1 :]
            if remaining:
                raise ValueError("Moves after double pass are not allowed")
            break
    canonical_moves = [PASS_TOKEN if move in ("pa", "pass") else move for move in moves]
    if legal_moves(board, "B") or legal_moves(board, "W"):
        raise ValueError("Othello game is not terminal")
    return canonical_moves


def _parse_headers(block):
    headers = {}
    for line in block.splitlines():
        match = re.match(r'\[(\w+)\s+"([^"]*)"\]', line.strip())
        if match:
            headers[match.group(1).lower()] = match.group(2)
    return headers


def _strip_headers_and_comments(block):
    body_lines = [line for line in block.splitlines() if not line.strip().startswith("[")]
    body = "\n".join(body_lines)
    body = re.sub(r"\{[^}]*\}", " ", body)
    body = re.sub(r";[^\n]*", " ", body)
    return body


def _moves_from_pgn_body(body):
    body = re.sub(r"\d+\s*\.", " ", body)
    candidates = re.findall(r"\b(?:[A-Ha-h][1-8]|PA|pa|PASS|pass)\b", body)
    return [m.lower() for m in candidates]


def tokens_from_othello_moves(moves):
    canonical = validate_othello_moves([str(m).lower() for m in moves])
    if len(canonical) < 8:
        raise ValueError("Othello game is too short")
    return ["<bos>", "<othello>"] + canonical + ["<eos>"]


def parse_othello_pgn_to_tokens(pgn_path, max_games=None):
    """
    Parses Othello PGN/WTHOR-style text and yields legal move-only token streams.
    """
    if not os.path.exists(pgn_path):
        print(f"[Error] Othello PGN file not found: {pgn_path}")
        return

    print(f"[Parsing Othello] Reading games from {os.path.basename(pgn_path)}...")

    games_parsed = 0
    current = []
    opener = gzip.open if str(pgn_path).lower().endswith(".gz") else open
    with opener(pgn_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("[Event ") and current:
                yielded = _parse_othello_block("".join(current), pgn_path)
                if yielded is not None:
                    games_parsed += 1
                    yield yielded
                    if max_games and games_parsed >= max_games:
                        return
                current = []
            current.append(line)

    if current and (not max_games or games_parsed < max_games):
        yielded = _parse_othello_block("".join(current), pgn_path)
        if yielded is not None:
            games_parsed += 1
            yield yielded

    print(f"[Success] Parsed {games_parsed} Othello games.")


def _parse_othello_block(block, source_path):
    if not block.strip():
        return None
    headers = _parse_headers(block)
    moves = _moves_from_pgn_body(_strip_headers_and_comments(block))
    try:
        tokens = tokens_from_othello_moves(moves)
    except ValueError as exc:
        print(f"[Warning] Skipping illegal Othello game in {os.path.basename(source_path)}: {exc}")
        return None
    metadata = {
        "black": headers.get("black", "Unknown"),
        "white": headers.get("white", "Unknown"),
        "result": headers.get("result", "*"),
        "date": headers.get("date", "????"),
        "move_count": len(tokens) - 3,
        "source_path": str(Path(source_path).resolve()),
    }
    return tokens, metadata


def _int_to_square(value):
    value = int(value)
    if value < 0:
        return None
    if value in (64, 65):
        return PASS_TOKEN
    if not 0 <= value < 64:
        raise ValueError(f"Othello integer move out of range: {value}")
    return f"{chr(ord('a') + value % 8)}{value // 8 + 1}"


def _moves_from_row_values(values):
    moves = []
    for value in values:
        move = _int_to_square(value) if isinstance(value, int) else str(value).lower()
        if move in (None, "", "nan", "none", "pad"):
            continue
        moves.append(move)
    return moves


def parse_othello_jsonl_to_tokens(jsonl_path, max_games=None):
    parsed = 0
    opener = gzip.open if str(jsonl_path).lower().endswith(".gz") else open
    with opener(jsonl_path, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            row = json.loads(line)
            values = row.get("moves") or row.get("games") or row.get("seqs") or row.get("sequence")
            if values is None:
                continue
            moves = _moves_from_row_values(values)
            try:
                tokens = tokens_from_othello_moves(moves)
            except ValueError:
                continue
            parsed += 1
            yield tokens, {
                "source": "othello_jsonl",
                "filename": os.path.basename(jsonl_path),
                "row": line_number,
                "move_count": len(tokens) - 3,
                "source_path": str(Path(jsonl_path).resolve()),
            }
            if max_games and parsed >= max_games:
                break


def parse_othello_hf_dataset(dataset_id, split="train", max_games=None):
    from datasets import load_dataset

    parsed = 0
    ds = load_dataset(dataset_id, split=split, streaming=True)
    for row_number, row in enumerate(ds, 1):
        values = row.get("moves") or row.get("games") or row.get("seqs") or row.get("sequence")
        if values is None:
            continue
        moves = _moves_from_row_values(values)
        try:
            tokens = tokens_from_othello_moves(moves)
        except ValueError:
            continue
        parsed += 1
        yield tokens, {
            "source": dataset_id,
            "split": split,
            "row": row_number,
            "move_count": len(tokens) - 3,
        }
        if max_games and parsed >= max_games:
            break


def parse_othello_parquet_to_tokens(parquet_path, max_games=None):
    from datasets import load_dataset

    parsed = 0
    ds = load_dataset("parquet", data_files=str(parquet_path), split="train", streaming=True)
    for row_number, row in enumerate(ds, 1):
        values = row.get("moves") or row.get("games") or row.get("seqs") or row.get("sequence")
        if values is None:
            continue
        moves = _moves_from_row_values(values)
        try:
            tokens = tokens_from_othello_moves(moves)
        except ValueError:
            continue
        parsed += 1
        yield tokens, {
            "source": "othello_parquet",
            "filename": os.path.basename(parquet_path),
            "row": row_number,
            "move_count": len(tokens) - 3,
            "seat_count": 2,
            "view_type": "complete",
            "viewer_seat": None,
            "source_path": str(Path(parquet_path).resolve()),
        }
        if max_games and parsed >= max_games:
            break


def parse_othello_inputs(input_path, max_games=None):
    path = Path(input_path)
    files = []
    if path.is_file():
        files = [path]
    else:
        for root, dirs, names in os.walk(path):
            dirs.sort()
            for name in sorted(names):
                if name.lower().endswith((".pgn", ".pgn.gz", ".jsonl", ".jsonl.gz", ".parquet")):
                    files.append(Path(root) / name)
    parsed = 0
    for file_path in files:
        remaining = None if max_games is None else max_games - parsed
        if remaining is not None and remaining <= 0:
            break
        lower = file_path.name.lower()
        if lower.endswith((".jsonl", ".jsonl.gz")):
            iterator = parse_othello_jsonl_to_tokens(str(file_path), max_games=remaining)
        elif lower.endswith(".parquet"):
            iterator = parse_othello_parquet_to_tokens(str(file_path), max_games=remaining)
        else:
            iterator = parse_othello_pgn_to_tokens(str(file_path), max_games=remaining)
        for tokens, metadata in iterator:
            parsed += 1
            metadata.setdefault("seat_count", 2)
            metadata.setdefault("view_type", "complete")
            metadata.setdefault("viewer_seat", None)
            yield tokens, metadata
