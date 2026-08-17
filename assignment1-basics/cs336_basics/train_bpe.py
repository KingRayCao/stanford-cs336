from .pretokenization import pre_tokenization
from collections import Counter


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab: dict[int, bytes] = dict()
    merges: list[tuple[bytes, bytes]] = list()
    vocab_pair_counter: Counter[tuple[bytes, bytes]] = Counter()
    vocab2token: dict[tuple[bytes, bytes], set[bytes]] = dict()
    token2vocab: dict[bytes, list[bytes]] = dict()

    assert vocab_size >= len(special_tokens) + 256, "Vocab size must be at least 256 + number of special tokens"

    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode("utf-8")
    for i in range(256):
        vocab[i + len(special_tokens)] = bytes([i])

    if len(vocab) == vocab_size:
        return vocab, merges

    tokens = pre_tokenization(input_path, special_tokens, num_processes)
    print(f"Pre-tokenization complete. Found {len(tokens)} unique pre-tokens.")
    tokens = {token.encode("utf-8"): count for token, count in tokens.items()}
    # init
    best_pair = None
    best_count = 0
    for token, count in tokens.items():
        token2vocab[token] = [token[0:1]]
        for i in range(len(token) - 1):
            pair = (token[i : i + 1], token[i + 1 : i + 2])
            if pair not in vocab2token:
                vocab2token[pair] = set()
            vocab2token[pair].add(token)
            vocab_pair_counter[pair] += count
            token2vocab[token].append(token[i + 1 : i + 2])
    best_pair = find_best_pair(vocab_pair_counter)
    if best_pair is None:
        return vocab, merges
    merge_token_pair(best_pair, tokens, vocab2token, token2vocab, vocab_pair_counter, vocab, merges)

    for i in range(vocab_size - len(special_tokens) - 257):
        current_vocab_size = len(vocab)
        best_pair = find_best_pair(vocab_pair_counter)
        if best_pair is None:
            break

        if current_vocab_size % 100 == 0:
            print(f"Training BPE: {current_vocab_size} / {vocab_size}")
            print(f"    Current best pair: {best_pair} with count {vocab_pair_counter[best_pair]}")

        merge_token_pair(best_pair, tokens, vocab2token, token2vocab, vocab_pair_counter, vocab, merges)

    return vocab, merges


def merge_token_pair(
    pair: tuple[bytes, bytes],
    tokens: dict[bytes, int],
    vocab2token: dict[tuple[bytes, bytes], set[bytes]],
    token2vocab: dict[bytes, list[bytes]],
    vocab_pair_counter: Counter[tuple[bytes, bytes]],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> None:
    new_vocab = pair[0] + pair[1]
    tokens_to_merge = vocab2token[pair].copy()
    for token in tokens_to_merge:
        token_count = tokens[token]
        token_vocab = token2vocab[token]
        # undo the pair counts for the old pairs in the token
        for i in range(len(token_vocab) - 1):
            old_pair = (token_vocab[i], token_vocab[i + 1])
            vocab_pair_counter[old_pair] -= token_count
            if __debug__:
                assert vocab_pair_counter[old_pair] >= 0, (
                    f"merge_token_pair failed: count for pair {old_pair} is negative after decrementing by {token_count}"
                )
            if vocab_pair_counter[old_pair] == 0:
                del vocab_pair_counter[old_pair]
            vocab2token[old_pair].discard(token)
        new_token = []
        i = 0
        while i < len(token_vocab):
            if i < len(token_vocab) - 1 and (token_vocab[i], token_vocab[i + 1]) == pair:
                new_token.append(new_vocab)
                i += 2
            else:
                new_token.append(token_vocab[i])
                i += 1
        if __debug__:
            merged_token = b"".join(new_token)
            assert merged_token == token, (
                f"merge_token_pair failed: merged token {merged_token!r} does not match original token {token!r}"
            )
        token2vocab[token] = new_token
        # update new pair count
        for i in range(len(new_token) - 1):
            new_pair = (new_token[i], new_token[i + 1])
            vocab_pair_counter[new_pair] += token_count
            if new_pair not in vocab2token:
                vocab2token[new_pair] = set()
            vocab2token[new_pair].add(token)

    vocab[len(vocab)] = new_vocab
    merges.append(pair)


def find_best_pair(vocab_pair_counter: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
    best_pair = None
    best_count = 1
    for vocab_pair, count in vocab_pair_counter.items():
        if count > best_count:
            best_count = count
            best_pair = vocab_pair
        elif count == best_count:
            if best_pair is None or vocab_pair > best_pair:
                best_pair = vocab_pair
    return best_pair
