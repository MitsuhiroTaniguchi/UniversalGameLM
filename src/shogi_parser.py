import os
import lzma
import re
import tempfile
from pathlib import Path
import cshogi


TERMINAL_MARKERS = {
    "%TORYO": "resign",
    "%SENNICHITE": "repetition",
    "%JISHOGI": "impasse",
    "%TIME_UP": "time_up",
    "%ILLEGAL_MOVE": "illegal_move",
    "%KACHI": "declaration_win",
    "%HIKIWAKE": "draw",
    "%CHUDAN": "interrupted",
}


def _read_csa_text(csa_path):
    path = Path(csa_path)
    suffix_pair = "".join(path.suffixes[-2:]).lower()
    if suffix_pair == ".csa.xz" or path.suffix.lower() == ".xz":
        with lzma.open(csa_path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    with open(csa_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _terminal_reason(csa_text):
    for line in reversed(csa_text.splitlines()):
        marker = line.strip().split(",", 1)[0]
        if marker in TERMINAL_MARKERS:
            return TERMINAL_MARKERS[marker], marker
    return None, None


def _setup_tokens(csa_text):
    tokens = []
    explicit_position = False
    side_to_move = None
    for line in csa_text.splitlines():
        line = line.strip()
        if line == "+" or line == "-":
            side_to_move = "black" if line == "+" else "white"
        if line.startswith("PI"):
            continue
        if line.startswith("P") and not re.match(r"^P[1-9]", line):
            explicit_position = True
    if explicit_position:
        raise ValueError("Explicit CSA setup is not serialized; reject non-PI starts")
    if side_to_move:
        tokens.append(f"TURN:{side_to_move}")
    return tokens


def _collapse_explicit_standard_board(csa_text):
    lines = csa_text.splitlines()
    board_lines = [line for line in lines if re.match(r"^P[1-9]", line.strip())]
    if not board_lines:
        return csa_text
    collapsed = []
    inserted = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^P[1-9]", stripped):
            if not inserted:
                collapsed.append("PI")
                inserted = True
            continue
        collapsed.append(line)
    return "\n".join(collapsed) + ("\n" if csa_text.endswith("\n") else "")


def _date_from_csa(csa_text):
    for line in csa_text.splitlines():
        if line.startswith("$START_TIME:"):
            return line.split(":", 1)[1][:10].replace("/", ".").replace("-", ".")
    return None

def parse_csa_to_tokens(csa_path):
    """
    Parses a single Shogi CSA file using cshogi and returns the token sequence and metadata.
    Token sequence format: ['<bos>', '<shogi>', '7g7f', '1c1d', ..., '<eos>']
    """
    if not os.path.exists(csa_path):
        print(f"[Error] CSA file not found: {csa_path}")
        return None, None

    try:
        csa_text = _read_csa_text(csa_path)
        terminal_reason, terminal_marker = _terminal_reason(csa_text)
        if terminal_reason is None:
            return None, None
        if terminal_reason in {"illegal_move", "interrupted"}:
            return None, None

        parser = cshogi.Parser()
        parser.parse_csa_str(_collapse_explicit_standard_board(csa_text))
        
        # Convert move integers to USI strings
        usi_moves = [cshogi.move_to_usi(m) for m in parser.moves]
        if any(move in (None, "None", "") for move in usi_moves):
            return None, None
        
        # Quality Filter: Skip extremely short or empty games
        if len(usi_moves) < 10:
            return None, None

        tokens = ["<bos>", "<shogi>"] + _setup_tokens(csa_text) + usi_moves + [f"END:{terminal_reason}", "<eos>"]
        
        # Determine winner
        winner = "Draw"
        if parser.win == cshogi.BLACK_WIN:
            winner = "Black"
        elif parser.win == cshogi.WHITE_WIN:
            winner = "White"

        metadata = {
            "black": parser.names[0] if len(parser.names) > 0 else "Unknown",
            "white": parser.names[1] if len(parser.names) > 1 else "Unknown",
            "winner": winner,
            "terminal": terminal_reason,
            "terminal_marker": terminal_marker,
            "date": _date_from_csa(csa_text),
            "move_count": len(usi_moves),
            "filename": os.path.basename(csa_path),
            "source_path": str(Path(csa_path).resolve()),
        }
        
        return tokens, metadata
    except Exception as e:
        print(f"[Warning] Failed to parse {os.path.basename(csa_path)}: {e}")
        return None, None

def iter_csa_files(directory_path):
    path = Path(directory_path)
    if path.is_file():
        if path.name.lower().endswith((".csa", ".csa.xz")):
            yield str(path)
        elif path.name.lower().endswith(".7z"):
            try:
                import py7zr
            except ImportError as exc:
                raise RuntimeError("Reading .7z CSA archives requires py7zr") from exc
            with tempfile.TemporaryDirectory() as temp_dir:
                with py7zr.SevenZipFile(path, mode="r") as archive:
                    archive.extractall(path=temp_dir)
                yield from iter_csa_files(temp_dir)
        return

    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            if name.lower().endswith((".csa", ".csa.xz")):
                yield str(Path(root) / name)
            elif name.lower().endswith(".7z"):
                yield from iter_csa_files(Path(root) / name)


def parse_shogi_directory(directory_path, max_games=None):
    """
    Parses all CSA files in a directory and yields token sequences.
    """
    print(f"[Parsing Shogi] Streaming CSA files from {directory_path}...")
    
    games_parsed = 0
    for filepath in iter_csa_files(directory_path):
        tokens, metadata = parse_csa_to_tokens(filepath)
        if tokens is not None:
            games_parsed += 1
            yield tokens, metadata
            
            if max_games and games_parsed >= max_games:
                break
                
    print(f"[Success] Parsed {games_parsed} shogi games.")

if __name__ == "__main__":
    # Test parser
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_shogi_dir = os.path.join(base_dir, "data", "shogi")
    
    if os.path.exists(test_shogi_dir):
        # Find some files and parse them
        for i, (tokens, meta) in enumerate(parse_shogi_directory(test_shogi_dir, max_games=3)):
            print(f"\nGame #{i+1} Metadata: {meta}")
            print(f"Tokens (length {len(tokens)}): {tokens[:15]} ... {tokens[-5:]}")
    else:
        print("[Info] Please run src/download.py first to download the test Shogi logs.")
