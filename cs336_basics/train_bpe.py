# Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer. Your BPE training function should handle (at least) the following input parameters:

import regex as re  # `regex` supports \p{L}, \p{N} Unicode property escapes; stdlib `re` does not
import json
import multiprocessing
from tqdm import tqdm
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _pretokenize_chunk(args: tuple) -> dict[tuple, int]:
    """Worker function: pre-tokenize a single chunk and return deduplicated counts."""
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # Split the chunk around special tokens so their bytes don't participate in merging
    if special_tokens:
        special_pattern = "|".join(re.escape(t) for t in special_tokens)
        segments = re.split(special_pattern, chunk)
    else:
        segments = [chunk]

    counts: dict[tuple, int] = {}
    for segment in segments:
        for match in re.finditer(PAT, segment):
            pre_token_bytes = match.group(0).encode("utf-8")
            key = tuple(bytes([b]) for b in pre_token_bytes)
            counts[key] = counts.get(key, 0) + 1

    return counts

class BPETrainer:

    def __init__(self):
        pass

    def train(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        debug: bool = False,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Train a byte-level BPE tokenizer on the given input text file.
        Args:
            input_path: Path to a text file with BPE tokenizer training data.
            vocab_size: A positive integer that defines the maximum final vocabulary size (including the
                initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
            special_tokens: A list of strings to add to the vocabulary. These special tokens do not
                otherwise affect BPE training.
        Returns:
            vocab: The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
            merges: A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
        """

        # --- Step 1: Find chunk boundaries ---
        import time as _time
        t0 = _time.time()
        num_processes = multiprocessing.cpu_count()
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        if debug:
            with open(input_path, "r", encoding="utf-8") as f:
                print(f"First 100 characters of the input text: {f.read(100)}")
            # For debug, just use the first chunk boundary pair
            boundaries = boundaries[:2]

        # --- Step 2: Pre-tokenization ---
        # tiktoken-style regex splits text into "words" before byte-level BPE
        # PAT is defined at module level so the worker process can use it

        # --- Step 3: Build initial vocab ---
        # HINT: vocab starts with 256 single-byte tokens: {0: b'\x00', 1: b'\x01', ..., 255: b'\xff'}
        # HINT: then assign IDs to special_tokens (e.g. IDs 256, 257, ...)
        # HINT: vocab should NOT contain pre-tokens as entries; only bytes/merges/special tokens go in vocab
        vocab = {i: bytes([i]) for i in range(256)}
        special_token_ids = {}
        for i, token in enumerate(special_tokens):
            special_token_ids[token] = 256 + i
            vocab[256 + i] = token.encode("utf-8")

        # --- Step 4: Parallel pre-tokenization ---
        t1 = _time.time()
        print(f"[Step 1] Chunk boundaries: {t1 - t0:.2f}s")
        chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        if len(chunk_args) <= 1:
            chunk_results = [_pretokenize_chunk(args) for args in chunk_args]
        else:
            with multiprocessing.Pool(processes=num_processes) as pool:
                chunk_results = pool.map(_pretokenize_chunk, chunk_args)
        
        # flatten and deduplicate: merge counts from all chunks
        pre_token_counts: dict[tuple, int] = {}
        for chunk_counts in chunk_results:
            for key, count in chunk_counts.items():
                pre_token_counts[key] = pre_token_counts.get(key, 0) + count
        if debug:
            print(f"Chunk size: {boundaries[1] - boundaries[0]} bytes")
            print(f"Total unique pre-tokens: {len(pre_token_counts)}")

        # --- Step 5: Track merges ---
        merges = []

        next_vocab_id = len(vocab)

        # --- Step 6: BPE merge loop ---
        t2 = _time.time()
        print(f"[Step 2] Pre-tokenization + dedup: {t2 - t1:.2f}s")
        # Build initial freq from scratch, then update incrementally
        freq: dict[tuple[bytes, bytes], int] = {}
        for seq_tuple, count in pre_token_counts.items():
            for i in range(len(seq_tuple) - 1):
                pair = (seq_tuple[i], seq_tuple[i+1])
                freq[pair] = freq.get(pair, 0) + count

        # Build inverted index: pair -> set of seq_tuples that contain it
        pair_to_seqs: dict[tuple[bytes, bytes], set[tuple]] = {}
        for seq_tuple in pre_token_counts:
            for i in range(len(seq_tuple) - 1):
                pair = (seq_tuple[i], seq_tuple[i+1])
                if pair not in pair_to_seqs:
                    pair_to_seqs[pair] = set()
                pair_to_seqs[pair].add(seq_tuple)

        num_merges = vocab_size - len(vocab)
        t3 = _time.time()
        print(f"[Step 3] Build freq + inverted index: {t3 - t2:.2f}s")
        pbar = tqdm(total=num_merges, desc="BPE merges")
        while len(vocab) < vocab_size:
            if not freq:
                break

            # --- Step 6b: Find the most frequent pair ---
            most_frequent_pair = max(freq.keys(), key=lambda pair: (freq[pair], pair[0], pair[1]))

            if debug:
                print("ITERATION MERGING", next_vocab_id, most_frequent_pair)

            # --- Step 6c: Create the new merged token ---
            merged_token = most_frequent_pair[0] + most_frequent_pair[1]
            vocab[next_vocab_id] = merged_token
            next_vocab_id += 1
            merges.append(most_frequent_pair)
            pbar.update(1)

            # --- Step 6d: Apply merge and incrementally update freq ---
            # Only process sequences that actually contain the merged pair
            affected_seqs = pair_to_seqs.pop(most_frequent_pair, set())
            del freq[most_frequent_pair]

            for seq_tuple in affected_seqs:
                count = pre_token_counts.pop(seq_tuple)
                seq = list(seq_tuple)

                # Subtract old pair counts and remove from inverted index
                for i in range(len(seq) - 1):
                    p = (seq[i], seq[i+1])
                    if p != most_frequent_pair:
                        freq[p] = freq.get(p, 0) - count
                        if freq[p] <= 0:
                            del freq[p]
                            pair_to_seqs.pop(p, None)
                        elif p in pair_to_seqs:
                            pair_to_seqs[p].discard(seq_tuple)

                # Apply the merge
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i+1]) == most_frequent_pair:
                        new_seq.append(merged_token)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1

                new_seq_tuple = tuple(new_seq)

                # If this new sequence already exists, merge the counts
                if new_seq_tuple in pre_token_counts:
                    # Need to subtract old pairs for the existing entry first,
                    # then re-add with combined count
                    existing_count = pre_token_counts[new_seq_tuple]
                    for i in range(len(new_seq_tuple) - 1):
                        p = (new_seq_tuple[i], new_seq_tuple[i+1])
                        freq[p] = freq.get(p, 0) - existing_count
                        if freq[p] <= 0:
                            del freq[p]
                            pair_to_seqs.pop(p, None)
                        elif p in pair_to_seqs:
                            pair_to_seqs[p].discard(new_seq_tuple)
                    count += existing_count

                pre_token_counts[new_seq_tuple] = count

                # Add new pair counts and update inverted index
                for i in range(len(new_seq_tuple) - 1):
                    p = (new_seq_tuple[i], new_seq_tuple[i+1])
                    freq[p] = freq.get(p, 0) + count
                    if p not in pair_to_seqs:
                        pair_to_seqs[p] = set()
                    pair_to_seqs[p].add(new_seq_tuple)

        # --- Step 7: Return vocab and merges ---
        t4 = _time.time()
        print(f"[Step 4] Merge loop: {t4 - t3:.2f}s")
        pbar.close()
        return (vocab, merges)



if __name__ == "__main__":
    import time
    import tracemalloc

    tracemalloc.start()
    start_time = time.time()

    trainer = BPETrainer()
    vocab, merges = trainer.train(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Peak memory: {peak / 1024**3:.2f} GB")

    # Find longest token
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token} (length {len(longest_token)} bytes)")
    print(f"Longest token decoded: {longest_token.decode('utf-8', errors='replace')}")

    # Serialize vocab and merges
    vocab_serializable = {str(k): v.hex() for k, v in vocab.items()}
    with open("data/bpe_vocab.json", "w") as f:
        json.dump(vocab_serializable, f, indent=2)

    merges_serializable = [[m[0].hex(), m[1].hex()] for m in merges]
    with open("data/bpe_merges.json", "w") as f:
        json.dump(merges_serializable, f, indent=2)

    print(f"Vocab size: {len(vocab)}")
    print(f"Merges count: {len(merges)}")
    print("Saved vocab to data/bpe_vocab.json and merges to data/bpe_merges.json")
