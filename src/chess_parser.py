import os
import chess.pgn

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
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            # Read game from PGN
            game = chess.pgn.read_game(f)
            if game is None:
                break  # End of file

            # Convert moves to UCI tokens
            tokens = ["<bos>", "<chess>"]
            
            # Reconstruct board to iterate over mainline moves
            board = game.board()
            move_count = 0
            
            for move in game.mainline_moves():
                tokens.append(move.uci())
                move_count += 1

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
                "move_count": move_count
            }
            
            yield tokens, metadata

            if max_games and games_parsed >= max_games:
                break

    print(f"[Success] Parsed {games_parsed} chess games.")

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
