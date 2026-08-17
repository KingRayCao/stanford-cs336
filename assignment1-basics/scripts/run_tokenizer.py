import numpy as np
from multiprocessing import Pool, Value
import argparse

from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization import find_chunk_boundaries

CHECK = True
PROGRESS_COUNTER = None
NUM_PROCESSES = 8


def initialize_worker(progress_counter) -> None:
    global PROGRESS_COUNTER
    PROGRESS_COUNTER = progress_counter


def chunk_encode(task) -> np.ndarray:
    tokenizer, input_path, start, end = task
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
    lines = chunk.splitlines(keepends=True)

    def tracked_lines():
        for line in lines:
            yield line
            with PROGRESS_COUNTER.get_lock():
                PROGRESS_COUNTER.value += 1

    return np.fromiter(tokenizer.encode_iterable(tracked_lines()), dtype=np.uint16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BPE tokenizer.")
    parser.add_argument("--vocab_path", type=str, help="Path to the vocabulary JSON file.")
    parser.add_argument("--merges_path", type=str, help="Path to the merges JSON file.")
    parser.add_argument("--input_text_path", type=str, help="Input text to tokenize.")
    parser.add_argument("--output_path", type=str, help="Path to save the encoded output as a .npy file.")
    parser.add_argument("--special_tokens", nargs="+", default=["<|endoftext|>"], help="List of special tokens.")

    args = parser.parse_args()

    # Initialize the tokenizer
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, args.special_tokens)

    num_processes = NUM_PROCESSES

    with open(args.input_text_path, "rb") as f:
        total_lines = sum(1 for _ in f)
        boundaries = find_chunk_boundaries(f, desired_num_chunks=num_processes, split_special_token=b"<|endoftext|>")

    tasks = [(tokenizer, args.input_text_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    if not tasks:
        print("No tasks to process.")
        exit(0)
    num_processes = min(num_processes, len(tasks))

    progress_counter = Value("Q", 0)
    with (
        Pool(
            processes=num_processes,
            initializer=initialize_worker,
            initargs=(progress_counter,),
        ) as pool,
        tqdm(
            total=total_lines,
            desc="Encoding",
            unit="line",
        ) as progress,
    ):
        result = pool.map_async(chunk_encode, tasks)
        displayed_lines = 0

        while not result.ready():
            result.wait(timeout=0.1)
            processed_lines = progress_counter.value
            progress.update(processed_lines - displayed_lines)
            displayed_lines = processed_lines

        encoded_chunks = result.get()
        processed_lines = progress_counter.value
        progress.update(processed_lines - displayed_lines)

    encode = np.concatenate(encoded_chunks)

    # print compression ratio
    original_size = sum(end - start for start, end in zip(boundaries[:-1], boundaries[1:]))
    compressed_size = encode.nbytes
    compression_ratio = original_size / compressed_size
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}")

    np.save(args.output_path, encode)

    if CHECK:
        # Check the decoding
        decoded = tokenizer.decode(encode.tolist())
        with open(args.input_text_path, "r", encoding="utf-8") as f:
            original = f.read()
        if decoded != original:
            print("Decoded text does not match the original text.")
            with open("decoded_output.txt", "w", encoding="utf-8") as f:
                f.write(decoded)
            with open("original_output.txt", "w", encoding="utf-8") as f:
                f.write(original)
            exit(1)
        else:
            print("Decoded text matches the original text.")
