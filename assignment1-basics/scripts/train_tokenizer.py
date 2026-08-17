def export_vocab_and_merges(
    vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_path: str, merges_path: str
) -> None:
    import json

    # Export vocab to JSON
    with open(vocab_path, "w", encoding="utf-8") as vocab_file:
        json.dump(
            [{"id": k, "hex": v.hex(), "text": v.decode("utf-8", errors="backslashreplace")} for k, v in vocab.items()],
            vocab_file,
            ensure_ascii=False,
            indent=2,
        )

    # Export merges to text file
    with open(merges_path, "w", encoding="utf-8") as merges_file:
        json.dump(
            [
                {
                    "id": i,
                    "pair_hex": (merge[0].hex(), merge[1].hex()),
                    "pair_text": (
                        merge[0].decode("utf-8", errors="backslashreplace"),
                        merge[1].decode("utf-8", errors="backslashreplace"),
                    ),
                }
                for i, merge in enumerate(merges)
            ],
            merges_file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    import argparse
    from cs336_basics.train_bpe import train_bpe

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--input_path", type=str, help="Path to the input text file.")
    parser.add_argument("--vocab_size", type=int, help="Desired vocabulary size.")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processes for pre tokenization.")
    parser.add_argument(
        "--special_tokens",
        nargs="+",
        default=["<|endoftext|>"],
        help="List of special tokens to include in the vocabulary.",
    )
    parser.add_argument(
        "--output_vocab_path", type=str, default="vocab.json", help="Path to save the vocabulary JSON file."
    )
    parser.add_argument(
        "--output_merges_path", type=str, default="merges.json", help="Path to save the merges JSON file."
    )

    args = parser.parse_args()

    vocab, merges = train_bpe(args.input_path, args.vocab_size, args.special_tokens)
    export_vocab_and_merges(vocab, merges, args.output_vocab_path, args.output_merges_path)
