import gzip
import os
from pathlib import Path

import chess
import chess.pgn


def _open_text(path):
    suffixes = Path(path).suffixes
    if suffixes[-1:] == [".gz"]:
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    if suffixes[-1:] == [".zst"]:
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "Reading .pgn.zst requires the optional zstandard package"
            ) from exc
        raw = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        return _ClosingTextWrapper(reader, raw)
    return open(path, "r", encoding="utf-8", errors="ignore")


class _ClosingTextWrapper:
    def __init__(self, reader, raw):
        import io

        self.raw = raw
        self.reader = reader
        self.text = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

    def __enter__(self):
        return self.text

    def __exit__(self, exc_type, exc, tb):
        self.text.close()
        self.reader.close()
        self.raw.close()


def iter_chess_inputs(input_path):
    path = Path(input_path)
    if path.is_dir():
        for root, _, files in os.walk(path):
            for name in sorted(files):
                lower = name.lower()
                if lower.endswith((".pgn", ".pgn.gz", ".pgn.zst")):
                    yield str(Path(root) / name)
    else:
        yield str(path)


def _game_context_tokens(game):
    tokens = []
    variant = game.headers.get("Variant")
    setup = game.headers.get("SetUp")
    fen = game.headers.get("FEN")
    if variant and variant != "?":
        tokens.append(f"VARIANT:{variant.lower().replace(' ', '_')}")
    if setup == "1" or fen:
        if not fen:
            fen = chess.STARTING_FEN
        tokens.append(f"FEN:{fen.replace(' ', '_')}")
    return tokens

def parse_pgn_to_tokens(pgn_path, max_games=None):
    """
    Parses a Chess PGN file and yields token sequences for each game.
    Token sequence format: ['<bos>', '<chess>', 'e2e4', 'e7e5', ..., '<eos>']
    """
    if not os.path.exists(pgn_path):
        print(f"[Error] PGN file not found: {pgn_path}")
        return

    print(f"[Parsing Chess] Reading games from {os.path.basename(pgn_path)}...")
    
    games_parsed = 0
    with _open_text(pgn_path) as f:
        while True:
            # Read game from PGN
            game = chess.pgn.read_game(f)
            if game is None:
                break  # End of file
            if game.errors:
                continue

            # Convert moves to UCI tokens
            context_tokens = _game_context_tokens(game)
            tokens = ["<bos>", "<chess>"] + context_tokens
            
            # Reconstruct board to iterate over mainline moves
            board = game.board()
            move_count = 0
            
            legal = True
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    legal = False
                    break
                tokens.append(move.uci())
                board.push(move)
                move_count += 1
            if not legal:
                continue

            tokens.append("<eos>")

            # Quality Filter: Skip extremely short or empty games
            if move_count < 4:
                continue

            games_parsed += 1
            
            # Yield game metadata along with tokens
            metadata = {
                "white": game.headers.get("White", "Unknown"),
                "black": game.headers.get("Black", "Unknown"),
                "white_elo": game.headers.get("WhiteElo", "?"),
                "black_elo": game.headers.get("BlackElo", "?"),
                "result": game.headers.get("Result", "*"),
                "date": game.headers.get("Date", "????.??.??"),
                "move_count": move_count,
                "variant": game.headers.get("Variant", "standard"),
                "fen": game.headers.get("FEN") if game.headers.get("SetUp") == "1" else None,
                "source_path": str(Path(pgn_path).resolve()),
            }
            
            yield tokens, metadata

            if max_games and games_parsed >= max_games:
                break

    print(f"[Success] Parsed {games_parsed} chess games.")


def parse_chess_inputs(input_path, max_games=None):
    parsed = 0
    for pgn_path in iter_chess_inputs(input_path):
        remaining = None if max_games is None else max_games - parsed
        if remaining is not None and remaining <= 0:
            break
        for tokens, metadata in parse_pgn_to_tokens(pgn_path, max_games=remaining):
            parsed += 1
            yield tokens, metadata

if __name__ == "__main__":
    # Test parser
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_pgn = os.path.join(base_dir, "data", "chess", "Carlsen.pgn")
    
    if os.path.exists(test_pgn):
        for i, (tokens, meta) in enumerate(parse_pgn_to_tokens(test_pgn, max_games=3)):
            print(f"\nGame #{i+1} Metadata: {meta}")
            print(f"Tokens (length {len(tokens)}): {tokens[:15]} ... {tokens[-5:]}")
    else:
        print("[Info] Please run src/download.py first to download the test PGN file.")
