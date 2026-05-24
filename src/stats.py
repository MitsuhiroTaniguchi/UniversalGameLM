import collections

def analyze_dataset_stats(dataset):
    """
    Analyzes and prints statistical insights for a tokenized dataset.
    dataset is a list of dicts: [{'tokens': [...], 'game': 'chess'/'shogi', 'metadata': {...}}]
    """
    by_game = collections.defaultdict(list)
    for item in dataset:
        by_game[item["game"]].append(item)

    print("\n" + "="*50)
    print("           UNIVERSAL GAME LM DATASET STATS")
    print("="*50)
    
    total_games = len(dataset)
    print(f"Total Games in Dataset: {total_games}")
    
    for game_name, games in by_game.items():
        count = len(games)
        print(f"\n--- {game_name.upper()} ({count} games) ---")
        
        # Sequence lengths
        lengths = [len(g["tokens"]) for g in games]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        
        # Sort to find median
        sorted_lengths = sorted(lengths)
        median_len = sorted_lengths[len(sorted_lengths)//2] if sorted_lengths else 0
        
        print(f"  Sequence Lengths:")
        print(f"    Min:    {min_len} tokens")
        print(f"    Max:    {max_len} tokens")
        print(f"    Avg:    {avg_len:.1f} tokens")
        print(f"    Median: {median_len} tokens")
        
        # Most common moves
        move_counts = collections.Counter()
        for g in games:
            # Skip special tokens <bos>, <eos>, <chess>, <shogi>
            moves = [t for t in g["tokens"] if not (t.startswith("<") and t.endswith(">"))]
            move_counts.update(moves)
            
        print(f"  Unique Moves: {len(move_counts)}")
        print(f"  Top 10 Most Common Moves:")
        for move, freq in move_counts.most_common(10):
            percent = (freq / sum(move_counts.values())) * 100 if move_counts else 0
            print(f"    {move:<8} : {freq:>5} times ({percent:>5.2f}%)")
            
    print("="*50 + "\n")

if __name__ == "__main__":
    # Test statistics
    mock_dataset = [
        {"tokens": ["<bos>", "<chess>", "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "<eos>"], "game": "chess"},
        {"tokens": ["<bos>", "<chess>", "e2e4", "c7c5", "g1f3", "d7d6", "<eos>"], "game": "chess"},
        {"tokens": ["<bos>", "<shogi>", "7g7f", "1c1d", "2g2f", "3c3d", "8h2b+", "<eos>"], "game": "shogi"}
    ]
    analyze_dataset_stats(mock_dataset)
