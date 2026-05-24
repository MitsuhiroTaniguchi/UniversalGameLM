import os
import glob
import cshogi

def parse_csa_to_tokens(csa_path):
    """
    Parses a single Shogi CSA file using cshogi and returns the token sequence and metadata.
    Token sequence format: ['<bos>', '<shogi>', '7g7f', '1c1d', ..., '<eos>']
    """
    if not os.path.exists(csa_path):
        print(f"[Error] CSA file not found: {csa_path}")
        return None, None

    try:
        parser = cshogi.Parser()
        parser.parse_csa_file(csa_path)
        
        # Convert move integers to USI strings
        usi_moves = [cshogi.move_to_usi(m) for m in parser.moves]
        
        # Quality Filter: Skip extremely short or empty games
        if len(usi_moves) < 10:
            return None, None

        tokens = ["<bos>", "<shogi>"] + usi_moves + ["<eos>"]
        
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
            "move_count": len(usi_moves),
            "filename": os.path.basename(csa_path)
        }
        
        return tokens, metadata
    except Exception as e:
        print(f"[Warning] Failed to parse {os.path.basename(csa_path)}: {e}")
        return None, None

def parse_shogi_directory(directory_path, max_games=None):
    """
    Parses all CSA files in a directory and yields token sequences.
    """
    csa_pattern = os.path.join(directory_path, "**", "*.csa")
    csa_files = glob.glob(csa_pattern, recursive=True)
    
    print(f"[Parsing Shogi] Found {len(csa_files)} CSA files in {directory_path}...")
    
    games_parsed = 0
    for filepath in csa_files:
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
