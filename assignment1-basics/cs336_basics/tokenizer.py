from collections import Counter
from typing import Iterable, Iterator
import regex as re
class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        # self.merges = merges
        self.merge_rank = {merge: rank for rank, merge in enumerate(merges)}
        if special_tokens is not None:
            special_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True)
        self.special_tokens = special_tokens
        self.token_encoded = {token: [token_id] for token_id, token in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        import json
        merges = []
        with open(vocab_filepath, "r") as vocab_file:
            vocab = {int(record["id"]): bytes.fromhex(record["hex"]) for record in json.load(vocab_file)}
        with open(merges_filepath, "r") as merges_file:
            merges = [(bytes.fromhex(merge["pair_hex"][0]), bytes.fromhex(merge["pair_hex"][1])) for merge in json.load(merges_file)]
        return cls(vocab, merges, special_tokens)

    def encode_token(self, token: str) -> list[int]:
        token_bytes = token.encode("utf-8")
        token_encoded = []
        if token_bytes in self.token_encoded:
            return self.token_encoded[token_bytes]
        else:
            # merge in the merges order
            token_vocabs = [token_bytes[i:i+1] for i in range(len(token_bytes))]
            # for merge in self.merges:
            #     new_token_vocabs = []
            #     i = 0
            #     while i < len(token_vocabs):
            #         if i < len(token_vocabs) - 1 and (token_vocabs[i], token_vocabs[i + 1]) == merge:
            #             new_token_vocabs.append(merge[0] + merge[1])
            #             i += 2
            #         else:
            #             new_token_vocabs.append(token_vocabs[i])
            #             i += 1
            #     token_vocabs = new_token_vocabs
            while(True):
                merge_pos, merge_rank = -1, float('inf')
                for i in range(len(token_vocabs) - 1):
                    pair = (token_vocabs[i], token_vocabs[i + 1])
                    if pair in self.merge_rank and self.merge_rank[pair] < merge_rank:
                        merge_pos, merge_rank = i, self.merge_rank[pair]
                if merge_pos != -1:
                    token_vocabs = token_vocabs[:merge_pos] + [token_vocabs[merge_pos] + token_vocabs[merge_pos + 1]] + token_vocabs[merge_pos + 2:]
                else:
                    break
            for vocab in token_vocabs:
                assert vocab in self.token_encoded, f"Token {vocab} not found in vocab"
                token_encoded.extend(self.token_encoded[vocab])
            self.token_encoded[token_bytes] = token_encoded
            return token_encoded

    def encode_ordinary_text(self, text: str) -> list[int]:
        from cs336_basics.pretokenization import PAT
        tokens_encoded = []
        for token in re.finditer(PAT, text):
            token_bytes = token.group(0).encode("utf-8")
            tokens_encoded.extend(self.encode_token(token.group(0)))
        return tokens_encoded

    def encode(self, text: str) -> list[int]:
        tokens_encoded = []
        cursor = 0
        if self.special_tokens:
            ST_PAT = r"|".join(re.escape(token) for token in self.special_tokens)
            for match in re.finditer(ST_PAT, text):
                if match.start() > cursor:
                    tokens_encoded.extend(self.encode_ordinary_text(text[cursor:match.start()]))
                tokens_encoded.extend(self.encode_token(match.group(0)))
                cursor = match.end()
        if cursor < len(text):
            tokens_encoded.extend(self.encode_ordinary_text(text[cursor:]))
        return tokens_encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return decoded_bytes.decode("utf-8", errors="replace")