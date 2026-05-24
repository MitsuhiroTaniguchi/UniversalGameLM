import os
import json
from src.download import download_chess_pgn, download_shogi_daily, download_go_sgf, download_othello_pgn
from src.chess_parser import parse_pgn_to_tokens
from src.shogi_parser import parse_shogi_directory
from src.go_parser import parse_go_directory
from src.othello_parser import parse_othello_pgn_to_tokens
from src.poker_parser import generate_poker_dataset
from src.tokenizer import UniversalGameTokenizer
from src.stats import analyze_dataset_stats

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZED_DIR = os.path.join(BASE_DIR, "tokenized_data")
os.makedirs(TOKENIZED_DIR, exist_ok=True)

DATASET_PATH = os.path.join(TOKENIZED_DIR, "dataset.jsonl")
VOCAB_PATH = os.path.join(TOKENIZED_DIR, "vocab.json")

def main():
    print("="*70)
    print("      UniversalGameLM Shared Dataset Orchestration Pipeline")
    print("="*70)
    
    # 1. Download Datasets
    print("\n--- Step 1: Downloading game records ---")
    download_chess_pgn("Carlsen")
    download_shogi_daily("2026/05/23", max_games=100)
    download_go_sgf(max_games=100)
    download_othello_pgn(max_years=1)
    
    # 2. Parse Chess Games
    print("\n--- Step 2: Parsing Chess PGN to UCI Tokens ---")
    chess_pgn_path = os.path.join(BASE_DIR, "data", "chess", "Carlsen.pgn")
    chess_games = []
    if os.path.exists(chess_pgn_path):
        for tokens, meta in parse_pgn_to_tokens(chess_pgn_path, max_games=100):
            chess_games.append({
                "game": "chess",
                "tokens": tokens,
                "metadata": meta
            })
    print(f"Loaded {len(chess_games)} chess games.")
    
    # 3. Parse Shogi Games
    print("\n--- Step 3: Parsing Shogi CSA to USI Tokens ---")
    shogi_dir = os.path.join(BASE_DIR, "data", "shogi")
    shogi_games = []
    if os.path.exists(shogi_dir):
        for tokens, meta in parse_shogi_directory(shogi_dir, max_games=100):
            shogi_games.append({
                "game": "shogi",
                "tokens": tokens,
                "metadata": meta
            })
    print(f"Loaded {len(shogi_games)} shogi games.")
    
    # 4. Parse Go Games
    print("\n--- Step 4: Parsing Go SGF to Two-Letter Coordinate Tokens ---")
    go_dir = os.path.join(BASE_DIR, "data", "go")
    go_games = []
    if os.path.exists(go_dir):
        for tokens, meta in parse_go_directory(go_dir, max_games=100):
            go_games.append({
                "game": "go",
                "tokens": tokens,
                "metadata": meta
            })
    print(f"Loaded {len(go_games)} Go games.")
    
    # 5. Parse Othello Games
    print("\n--- Step 5: Parsing Othello WTHOR PGN to Lowercase Coordinates ---")
    othello_pgn_path = os.path.join(BASE_DIR, "data", "othello", "WTH_2024.pgn")
    othello_games = []
    if os.path.exists(othello_pgn_path):
        for tokens, meta in parse_othello_pgn_to_tokens(othello_pgn_path, max_games=100):
            othello_games.append({
                "game": "othello",
                "tokens": tokens,
                "metadata": meta
            })
    print(f"Loaded {len(othello_games)} Othello games.")
    
    # 6. Parse Poker Games
    print("\n--- Step 6: Generating Texas Hold'em Action/State Tokens ---")
    poker_games = []
    for tokens, meta in generate_poker_dataset(n_hands=100):
        poker_games.append({
            "game": "poker",
            "tokens": tokens,
            "metadata": meta
        })
    print(f"Generated {len(poker_games)} simulated Poker hands.")
    
    # 7. Combine & Setup Tokenizer
    combined_games = chess_games + shogi_games + go_games + othello_games + poker_games
    if not combined_games:
        print("[Error] No games parsed. Exiting pipeline.")
        return
        
    print("\n--- Step 7: Building Shared Universal Vocabulary ---")
    # Define expanded special tokens
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<chess>", "<shogi>", "<go>", "<othello>", "<poker>"]
    tokenizer = UniversalGameTokenizer(special_tokens=special_tokens)
    
    all_token_lists = [g["tokens"] for g in combined_games]
    tokenizer.build_vocab(all_token_lists)
    tokenizer.save_vocab(VOCAB_PATH)
    
    # 8. Encode & Serialize Dataset
    print("\n--- Step 8: Serializing Unified Dataset ---")
    dataset_entries = []
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for game in combined_games:
            encoded_ids = tokenizer.encode(game["tokens"])
            entry = {
                "game": game["game"],
                "tokens": game["tokens"],
                "ids": encoded_ids,
                "metadata": game["metadata"]
            }
            dataset_entries.append(entry)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Successfully serialized {len(dataset_entries)} tokenized games to {DATASET_PATH}")
    
    # 9. Statistical Report
    print("\n--- Step 9: Compiling Expanded Dataset Statistics ---")
    analyze_dataset_stats(dataset_entries)
    
    print("\n" + "="*70)
    print("      UniversalGameLM Shared Dataset Orchestration Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
