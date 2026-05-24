import os
import re

def parse_othello_pgn_to_tokens(pgn_path, max_games=None):
    """
    Parses an Othello PGN file and yields token sequences for each game.
    Token sequence format: ['<bos>', '<othello>', 'f5', 'd6', ..., '<eos>']
    """
    if not os.path.exists(pgn_path):
        print(f"[Error] Othello PGN file not found: {pgn_path}")
        return

    print(f"[Parsing Othello] Reading games from {os.path.basename(pgn_path)}...")
    
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split games by double newlines or PGN headers [Event ...]
    game_blocks = re.split(r'\n(?=\[Event\s+)', content)
    
    games_parsed = 0
    for block in game_blocks:
        if not block.strip():
            continue
            
        # Parse headers
        headers = {}
        for line in block.split("\n"):
            match = re.match(r'\[(\w+)\s+"([^"]*)"\]', line)
            if match:
                headers[match.group(1).lower()] = match.group(2)
                
        # Parse moves using case-insensitive regex for squares A1-H8 or PA/PASS
        # FFO WTHOR PGN format represents moves as e.g. "1. F5 D6"
        moves_raw = re.findall(r'\b(?:[A-H][1-8]|[a-h][1-8]|PA|pa|pass|PASS)\b', block)
        
        # Convert all moves to lowercase coordinates
        moves = [m.lower() for m in moves_raw]
        
        # Quality Filter: Skip empty or invalid games
        if len(moves) < 8:
            continue

        tokens = ["<bos>", "<othello>"] + moves + ["<eos>"]
        
        metadata = {
            "black": headers.get("black", "Unknown"),
            "white": headers.get("white", "Unknown"),
            "result": headers.get("result", "*"),
            "date": headers.get("date", "????"),
            "move_count": len(moves)
        }
        
        yield tokens, metadata
        games_parsed += 1
        
        if max_games and games_parsed >= max_games:
            break

    print(f"[Success] Parsed {games_parsed} Othello games.")

if __name__ == "__main__":
    # Test parser
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_pgn = os.path.join(base_dir, "data", "othello", "WTH_2024.pgn")
    
    if os.path.exists(test_pgn):
        for i, (tokens, meta) in enumerate(parse_othello_pgn_to_tokens(test_pgn, max_games=3)):
            print(f"\nGame #{i+1} Metadata: {meta}")
            print(f"Tokens (length {len(tokens)}): {tokens[:15]} ... {tokens[-5:]}")
    else:
        print("[Info] Please run src/download.py first to download the test Othello PGN.")
