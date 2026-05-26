import collections

NON_MOVE_TOKENS = {
    "pk:act:preflop", "pk:act:flop", "pk:act:turn", "pk:act:river",
    "pk:act:deal_board", "pk:act:hidden",
    "view:complete", "view:omniscient",
    "game_start", "game_end", "round_start", "round_end",
}

NON_MOVE_PREFIXES = (
    "view:imperfect:",
    "ch:rule:", "ch:fen:",
    "sh:turn:", "sh:end:",
    "go:sz:", "go:km:", "go:ru:", "go:ha:",
    "go:setup_b", "go:setup_w", "go:setup_e",
    "mj:rule:", "mj:bakaze:", "mj:kyoku:", "mj:honba", "mj:riichi_sticks",
    "mj:tenbo:", "mj:dora", "mj:ura_dora", "mj:wall",
    "mj:haipai:", "mj:hidden_haipai:",
    "br:dealer:", "br:vul:", "br:contract:", "br:declarer:", "br:play_leader:", "br:trump:",
    "br:hand:",
    "pk:private_card", "pk:undealt_card",
    "pk:seat:", "pk:card", "pk:showdown:", "pk:winner:",
    "pk:VARIANT:", "pk:STARTING_STACKS:", "pk:BLINDS_OR_STRADDLES:", "pk:ANTES:", "pk:MIN_BET:",
    "pk:ANTE_TRIMMING_STATUS:", "pk:BETTING_TYPE:",
    "pk:num:", "pk:amt:",
)


_BRIDGE_BID_SEATS = frozenset({"br:bid:N", "br:bid:E", "br:bid:S", "br:bid:W"})


def _is_move_subtok(token):
    """Decomposed sub-token that completes a multi-token game action."""
    if len(token) == 4 and token[:3] in ("br:", "pk:"):
        return True
    if token in _BRIDGE_BID_SEATS:
        return True
    if token.startswith("ch:") and not token.startswith(("ch:w:", "ch:b:")):
        return True
    if token.startswith("sh:") and not token.startswith(("sh:b:", "sh:w:")):
        return True
    if token.startswith("go:") and token not in ("go:b", "go:w"):
        return True
    return False


def is_counted_move_token(token):
    if token.startswith("<") and token.endswith(">"):
        return False
    if token in NON_MOVE_TOKENS:
        return False
    if token.startswith(NON_MOVE_PREFIXES):
        return False
    return not _is_move_subtok(token)


class DatasetStatsAccumulator:
    """Tracks dataset statistics without retaining every serialized row."""
    def __init__(self):
        self.game_counts = collections.Counter()
        self.token_counts = collections.Counter()
        self.non_special_token_counts = collections.Counter()
        self.length_counts = collections.defaultdict(collections.Counter)
        self.move_counts = collections.defaultdict(collections.Counter)
        self.seat_count_rows = collections.defaultdict(collections.Counter)
        self.seat_count_tokens = collections.defaultdict(collections.Counter)
        self.seat_count_views = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))

    def update(self, item):
        game = item["game"]
        tokens = item["tokens"]
        metadata = item.get("metadata") or {}
        self.game_counts[game] += 1
        self.token_counts[game] += len(tokens)
        self.length_counts[game][len(tokens)] += 1

        moves = [t for t in tokens if is_counted_move_token(t)]
        self.non_special_token_counts[game] += len(moves)
        self.move_counts[game].update(moves)
        seat_count = metadata.get("seat_count")
        if seat_count is not None:
            seat_count = int(seat_count)
            self.seat_count_rows[game][seat_count] += 1
            self.seat_count_tokens[game][seat_count] += len(tokens)
            view_type = metadata.get("view_type") or "unknown"
            self.seat_count_views[game][seat_count][view_type] += 1

    def _median_length(self, game):
        total = self.game_counts[game]
        if total == 0:
            return 0

        midpoint = total // 2
        seen = 0
        for length, count in sorted(self.length_counts[game].items()):
            seen += count
            if seen > midpoint:
                return length
        return 0

    def summary(self, target_tokens_per_game=None):
        games = {}
        for game in sorted(self.game_counts):
            count = self.game_counts[game]
            token_count = self.token_counts[game]
            lengths = self.length_counts[game]
            avg_len = token_count / count if count else 0
            target = target_tokens_per_game or 0
            deficit = max(target - token_count, 0) if target else 0
            coverage = (token_count / target) if target else None
            games[game] = {
                "games": count,
                "tokens": token_count,
                "target_tokens": target or None,
                "token_deficit": deficit if target else None,
                "coverage": coverage,
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "avg_length": avg_len,
                "median_length": self._median_length(game),
                "unique_non_special_tokens": len(self.move_counts[game]),
                "non_special_tokens": self.non_special_token_counts[game],
                "top_non_special_tokens": self.move_counts[game].most_common(10),
            }
            if self.seat_count_rows.get(game):
                games[game]["by_seat_count"] = {
                    str(seat_count): {
                        "rows": self.seat_count_rows[game][seat_count],
                        "tokens": self.seat_count_tokens[game][seat_count],
                        "views": dict(sorted(self.seat_count_views[game][seat_count].items())),
                    }
                    for seat_count in sorted(self.seat_count_rows[game])
                }
        return games

    def print_report(self, target_tokens_per_game=None):
        print_dataset_stats(self.summary(target_tokens_per_game))

def print_dataset_stats(summary):
    total_games = sum(item["games"] for item in summary.values())
    total_tokens = sum(item["tokens"] for item in summary.values())

    print("\n" + "="*50)
    print("           UNIVERSAL GAME LM DATASET STATS")
    print("="*50)
    print(f"Total Games in Dataset: {total_games}")
    print(f"Total Tokens in Dataset: {total_tokens}")

    for game_name, item in summary.items():
        print(f"\n--- {game_name.upper()} ({item['games']} games) ---")
        print(f"  Tokens: {item['tokens']:,}")
        if item.get("target_tokens"):
            coverage = item["coverage"] or 0
            print(f"  Target: {item['target_tokens']:,} tokens")
            print(f"  Coverage: {coverage:.8%}")
            print(f"  Deficit: {item['token_deficit']:,} tokens")

        print(f"  Sequence Lengths:")
        print(f"    Min:    {item['min_length']} tokens")
        print(f"    Max:    {item['max_length']} tokens")
        print(f"    Avg:    {item['avg_length']:.1f} tokens")
        print(f"    Median: {item['median_length']} tokens")

        print(f"  Unique Non-Special Tokens: {item['unique_non_special_tokens']}")
        if item.get("by_seat_count"):
            print("  By Seat Count:")
            for seat_count, bucket in item["by_seat_count"].items():
                print(f"    {seat_count} seats: {bucket['rows']} rows, {bucket['tokens']:,} tokens, views={bucket['views']}")
        print(f"  Top 10 Non-Special Tokens:")
        total_non_special = item["non_special_tokens"] or 1
        for move, freq in item["top_non_special_tokens"]:
            percent = (freq / total_non_special) * 100
            print(f"    {move:<8} : {freq:>5} times ({percent:>5.2f}%)")

    print("="*50 + "\n")

def analyze_dataset_stats(dataset):
    """
    Analyzes and prints statistical insights for a tokenized dataset.
    dataset is a list of dicts: [{'tokens': [...], 'game': 'chess'/'shogi', 'metadata': {...}}]
    """
    accumulator = DatasetStatsAccumulator()
    for item in dataset:
        accumulator.update(item)
    accumulator.print_report()

if __name__ == "__main__":
    # Test statistics
    mock_dataset = [
        {"tokens": ["<bos>", "<chess>", "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "<eos>"], "game": "chess"},
        {"tokens": ["<bos>", "<chess>", "e2e4", "c7c5", "g1f3", "d7d6", "<eos>"], "game": "chess"},
        {"tokens": ["<bos>", "<shogi>", "7g7f", "1c1d", "2g2f", "3c3d", "8h2b+", "<eos>"], "game": "shogi"}
    ]
    analyze_dataset_stats(mock_dataset)
