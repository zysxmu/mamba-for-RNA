import sys
from pathlib import Path

VALID = set("ATCG")

def read_sequences(path: str):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".fa", ".fasta", ".fna"]:
        seqs = []
        cur = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur:
                        seqs.append("".join(cur))
                        cur = []
                else:
                    cur.append(line)
            if cur:
                seqs.append("".join(cur))
        return seqs

    else:
        # Treat other extensions as one-sequence-per-line text files.
        seqs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seqs.append(line)
        return seqs


def main(path: str):
    seqs = read_sequences(path)

    total = len(seqs)
    empty_count = 0
    bad_count = 0
    good_count = 0
    bad_examples = []

    for i, seq in enumerate(seqs):
        seq = seq.strip().upper()

        if len(seq) == 0:
            empty_count += 1
            bad_count += 1
            if len(bad_examples) < 10:
                bad_examples.append((i, seq, {"EMPTY"}))
            continue

        bad_chars = set(seq) - VALID
        if bad_chars:
            bad_count += 1
            if len(bad_examples) < 10:
                bad_examples.append((i, seq[:100], bad_chars))
        else:
            good_count += 1

    print(f"File: {path}")
    print(f"Total sequences: {total}")
    print(f"Valid sequences (ATCG only): {good_count}")
    print(f"Invalid sequences: {bad_count}")
    print(f"Empty sequences: {empty_count}")

    if bad_count == 0:
        print("Result: all sequences contain only ATCG.")
    else:
        print("Result: the dataset contains invalid sequences.")
        print("\nFirst 10 invalid examples:")
        for idx, seq, bad_chars in bad_examples:
            print(
                f"- Record {idx + 1}: invalid characters={sorted(bad_chars)} "
                f"| first 100 bases={seq}"
            )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_dna_dataset.py <data-file>")
        sys.exit(1)
    main(sys.argv[1])
