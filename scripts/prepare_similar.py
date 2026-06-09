import argparse
import os
from collections import defaultdict
from itertools import islice

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STOPWORDS_PATH = os.path.join(PROJECT_DIR, "files", "stopwords.txt")


# -----------------------
# Configuration
# -----------------------
NGRAM_PRIORITIES = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
MIN_CLUSTER_SIZE = 4

# -----------------------
# Utilities
# -----------------------
def tokenize(text):
    return text.lower().strip().split()

def load_stopwords(path_stopwords=DEFAULT_STOPWORDS_PATH):
    stopwords = set()
    with open(path_stopwords, mode="r", encoding="utf-8") as f:
        for line in f:
            token = line.strip().lower()
            if token and not token.startswith("#"):
                stopwords.add(token)
    return stopwords

def remove_stopwords(tokens, stopwords):
    return [t for t in tokens if t not in stopwords]

def ngrams(tokens, n):
    return zip(*(islice(tokens, i, None) for i in range(n)))


# -----------------------
# Main logic
# -----------------------
def assign_supcon_ids(
    df,
    min_ngram_coverage,
    text_col="transcript",
    ngram_priorities=NGRAM_PRIORITIES,
    min_cluster_size=MIN_CLUSTER_SIZE,
    stopwords=None,
):
    if stopwords is None:
        stopwords = load_stopwords()

    df = df.copy()
    df["supcon_id"] = -1
    df["supcon_ngram_n"] = -1

    next_cluster_id = 0
    assigned = set()

    for n in ngram_priorities:
        index = defaultdict(list)

        for i, text in df[text_col].items():
            if i in assigned:
                continue

            tokens = tokenize(text)
            content_tokens = remove_stopwords(tokens, stopwords)

            if len(content_tokens) < n:
                continue

            coverage = n / len(content_tokens)
            if coverage < min_ngram_coverage:
                continue

            for ng in ngrams(content_tokens, n):
                index[ng].append(i)

        for ng, rows in index.items():
            rows = [r for r in rows if r not in assigned]
            if len(rows) < min_cluster_size:
                continue

            for r in rows:
                df.at[r, "supcon_id"] = next_cluster_id
                df.at[r, "supcon_ngram_n"] = n
                assigned.add(r)

            next_cluster_id += 1

    return df


def coverage(value):
    value = float(value)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("coverage must be greater than 0 and at most 1")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create n-gram-based SupCon groups from a training CSV."
    )
    parser.add_argument(
        "--prefix",
        choices=["UWB_ATCOSIM", "UWB"],
        default="UWB_ATCOSIM",
        help="Dataset folder under files/ (default: UWB_ATCOSIM).",
    )
    parser.add_argument(
        "--coverage",
        type=coverage,
        default=0.4,
        help="Minimum n-gram coverage ratio (default: 0.4).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path_data = os.path.join(PROJECT_DIR, "files", args.prefix)
    path_input = os.path.join(path_data, "train.csv")
    path_output_dir = os.path.join(path_data, "generated")
    path_output = os.path.join(
        path_output_dir, f"ngram_{round(args.coverage * 100)}.csv"
    )

    df = pd.read_csv(path_input)
    df_out = assign_supcon_ids(df, text_col="transcript", min_ngram_coverage=args.coverage)

    print("=== Summary ===")
    print("Prefix:", args.prefix)
    print("Coverage:", args.coverage)
    print("Total samples:", len(df_out))
    print("Known:", (df_out.supcon_id >= 0).sum())
    print("Unknown:", (df_out.supcon_id == -1).sum())
    print("Groups:", df_out.supcon_id.nunique() - (1 if -1 in df_out.supcon_id.values else 0))

    os.makedirs(path_output_dir, exist_ok=True)
    df_out.to_csv(path_output, index=False)
    print("Wrote:", path_output)


if __name__ == "__main__":
    main()
