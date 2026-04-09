import regex as re
import json
from collections.abc import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = merges

        # Add special tokens to vocab if not already present
        self.special_tokens: list[str] = []
        if special_tokens:
            # Sort by length descending so longer special tokens match first
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            existing_bytes = set(self.vocab.values())
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in existing_bytes:
                    new_id = max(self.vocab.keys()) + 1
                    self.vocab[new_id] = token_bytes
                    existing_bytes.add(token_bytes)

        # Reverse vocab: bytes -> id
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}

        # Merge rank: (token1, token2) -> priority (lower = higher priority)
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath) as f:
            vocab_hex = json.load(f)
        vocab = {int(k): bytes.fromhex(v) for k, v in vocab_hex.items()}

        with open(merges_filepath) as f:
            merges_hex = json.load(f)
        merges = [(bytes.fromhex(m[0]), bytes.fromhex(m[1])) for m in merges_hex]

        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        """Apply BPE merges to a list of byte tokens, in merge priority order.
        This is the heart of encoding. Given a word split into individual bytes like [b'h', b'e', b'l', b'l', b'o']:

        1. Scan all adjacent pairs, look up each in merge_rank, find the one with lowest rank (lines 58–65)
        2. If no pairs have a merge rule, we're done — break (line 67–68) 
        3. Apply that merge everywhere in the sequence, greedy left-to-right with i+= 2 to avoid overlap (lines 70–81)
        4. Repeat — the while loop continues until no more merges apply
        """
        
        while len(tokens) >= 2:
            # Find the pair with the lowest merge rank
            best_pair = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # Apply this merge everywhere in the sequence
            merged = best_pair[0] + best_pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        ids = []

        # Split text around special tokens
        if self.special_tokens:
            special_pattern = "|".join(re.escape(t) for t in self.special_tokens)
            parts = re.split(f"({special_pattern})", text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue

            # Check if this part is a special token
            if part in self.special_tokens:
                part_bytes = part.encode("utf-8")
                ids.append(self.bytes_to_id[part_bytes])
                continue

            # Pre-tokenize with the GPT-2 regex pattern
            for match in re.finditer(PAT, part):
                word = match.group(0)
                # Convert to individual bytes
                tokens = [bytes([b]) for b in word.encode("utf-8")]
                # Apply BPE merges
                tokens = self._apply_merges(tokens)
                # Convert to IDs
                for token in tokens:
                    ids.append(self.bytes_to_id[token])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_strings = []
        for token_id in ids:
            byte_strings.append(self.vocab[token_id])
        return b"".join(byte_strings).decode("utf-8", errors="replace")
