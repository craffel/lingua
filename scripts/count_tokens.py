"""
Script to count tokens in JSONL files in parallel by chunk and segment. Each process will handle a specific segment of a chunk file, read only its assigned lines, tokenize the text, and count tokens and bytes. Results are aggregated by chunk and optionally by source.
The script uses a multiprocessing Pool to efficiently utilize CPU cores and can handle large files by processing them in segments without loading entire files into memory.
"""

# PYTHONPATH=/home/gsa/lingua:$PYTHONPATH python scripts/count_tokens.py --tokenizer_name huggingface --tokenizer_path  gpt2 --add_bos --add_eos --chunks 1 --workers=64 --segments_per_chunk=16
# PYTHONPATH=/home/gsa/lingua:$PYTHONPATH python scripts/count_tokens.py --tokenizer_name huggingface --tokenizer_path  meta-llama/Llama-3.2-1B --add_bos --add_eos --chunks 1 --workers=64 --segments_per_chunk=16
# PYTHONPATH=/home/gsa/lingua:$PYTHONPATH python scripts/count_tokens.py --tokenizer_name huggingface --tokenizer_path  google/gemma-2-2b --add_bos --add_eos --chunks 1 --workers=64 --segments_per_chunk=16

#  PYTHONPATH=$PROJECT/lingua:$PYTHONPATH python scripts/count_tokens.py config=count_tokens_hq_.yaml glob_pattern="arb_Arab*.jsonl"

## on vulcan
# PYTHONPATH=/home/gsa/lingua:$PYTHONPATH python scripts/count_tokens.py --tokenizer_name huggingface --tokenizer_path  meta-llama/Llama-3.2-1B --add_bos --add_eos --chunks 1 --workers=64 --segments_per_chunk=16 --input_path=/scratch/gsa/controlled_data
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parents[1].as_posix())
print(sys.path)
import argparse
import functools
import json
import logging
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Any, List, Optional

import yaml
from omegaconf import OmegaConf

from lingua.tokenizer import TokenizerArgs, build_tokenizer

# Configure logging to handle multiprocessing output cleanly
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(processName)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_file_line_offsets(file_path, num_segments):
    """
    Split a file into roughly equal segments by counting lines.
    Returns list of (start_line, end_line) tuples for each segment.
    """
    # First pass: count total lines
    with open(file_path, "r") as f:
        total_lines = sum(1 for _ in f)

    if total_lines == 0:
        return []

    lines_per_segment = max(1, total_lines // num_segments)
    segments = []

    for i in range(num_segments):
        start_line = i * lines_per_segment
        # Last segment gets all remaining lines
        end_line = total_lines if i == num_segments - 1 else (i + 1) * lines_per_segment

        if start_line < total_lines:
            segments.append((start_line, end_line))

    return segments


def process_file_segment(
    chunk_id, segment_id, start_line, end_line, file_path, args=None, tokenizer_args=None
):
    """
    Worker function to process a segment of a single chunk file.
    Each worker reads only its assigned line range.
    """
    start_time = time.time()

    # Re-build tokenizer inside the process
    tokenizer = build_tokenizer(
        tokenizer_args.name, tokenizer_args.path, tokenizer_args.tokenizers
    )
    logger.info(f"Tokenizer built, processing {file_path} segment {segment_id}")

    local_count = 0
    total_bytes = 0
    lines_processed = 0
    token_count_per_source = {}

    try:
        with open(file_path, "r") as f_in:
            for i, line in enumerate(f_in):
                # Skip lines outside our segment
                if i < start_line:
                    continue
                if end_line != float("inf") and i >= end_line:
                    break

                if not line.strip():
                    continue

                # Parse JSONL
                data = json.loads(line)
                text = data.get("text", "").encode("utf-8").decode("utf-8")

                # Tokenize
                tokens = tokenizer.encode(
                    text, add_bos=args.add_bos, add_eos=args.add_eos
                )
                local_count += len(tokens)
                total_bytes += len(text.encode("utf-8"))
                lines_processed += 1

                if "source" in data:
                    source = data["source"]
                    if source not in token_count_per_source:
                        token_count_per_source[source] = 0
                    token_count_per_source[source] += len(tokens)

                # Progress logging
                if lines_processed % 50000 == 0:
                    elapsed = time.time() - start_time
                    rate = lines_processed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Chunk {chunk_id} Segment {segment_id}: {lines_processed} lines, {local_count} tokens {total_bytes / 1e9:.2f} GB ({rate:.0f} lines/sec)"
                    )

        elapsed = time.time() - start_time
        logger.info(
            f"Chunk {chunk_id} Segment {segment_id} finished: {lines_processed} lines, {local_count} tokens {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s"
        )
        return (
            chunk_id,
            segment_id,
            local_count,
            total_bytes,
            token_count_per_source,
            file_path,
        )

    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id} segment {segment_id}: {e}")
        return (chunk_id, segment_id, 0, 0, {}, file_path)


import re


def extract_chunk_sequence_number(filename):
    """
    Extracts the chunk or sequence ID from three different filename formats.
    Returns the extracted number as an integer, or None if no match is found.
    """

    # 1. Pattern for ".chunk.XXX.jsonl" (Handles the first two formats)
    # Searches for ".chunk." followed by one or more digits (\d+),
    # and captures the digits.
    match_chunk = re.search(r"\.chunk\.(\d+)\.jsonl$", filename)
    if match_chunk:
        # Returns the captured digits (Group 1)
        return int(match_chunk.group(1))

    # 2. Pattern for "train_data_{chunk}-{seq}-8.jsonl"
    # Searches for "train_data_" followed by any characters, then a hyphen,
    # then "0" followed by the sequence number (\d), and captures the sequence number.
    # Note: This is specifically targeting the 'seq' number after the last hyphen before '-8.jsonl'.
    match_seq = re.search(r"train_data_.*-0(\d+)-8\.jsonl$", filename)
    if match_seq:
        # Returns the captured digits (Group 1)
        return int(match_seq.group(1))

    # If neither pattern is found
    return None


def create_work_items(chunks, input_path, glob_pattern, segments_per_chunk):
    """
    Create a list of work items (chunk_id, segment_id, start_line, end_line, file_path).
    Each chunk is split into multiple segments.
    """
    work_items = []

    files_to_process = [f for f in input_path.glob(glob_pattern) if "val" not in f.name]
    for file_path in files_to_process:
        chunk_id = extract_chunk_sequence_number(file_path.name)
        if chunk_id is None:
            logger.warning(f"Could not extract chunk ID from filename: {file_path.name}, will use None as key")
        if segments_per_chunk == 1:
            # Process entire file as single segment - skip line counting
            logger.info(
                f"Processing {file_path.name} as single segment (skipping line count)"
            )
            work_items.append((chunk_id, 0, 0, float("inf"), file_path))
        else:
            # Get line segments for this file
            logger.info(f"Analyzing {file_path.name} for segmentation...")
            segments = get_file_line_offsets(file_path, segments_per_chunk)

            logger.info(f"{file_path.name}: Split into {len(segments)} segments")

            for seg_id, (start_line, end_line) in enumerate(segments):
                work_items.append((chunk_id, seg_id, start_line, end_line, file_path))

    return work_items


def auto_detect_chunks(input_path):
    """Automatically detect available chunk files."""
    chunks = []
    for f in sorted(input_path.glob("train_data.*.jsonl")):
        try:
            chunk_num = int(f.stem.split(".")[-1])
            chunks.append(chunk_num)
        except ValueError:
            continue
    return chunks


@dataclass
class Args:
    tokenizer_name: Optional[str] = None # "huggingface"
    tokenizer_path: Optional[str] = None # "meta-llama/Llama-3.2-1B"
    tokenizer: Optional[TokenizerArgs] = None
    input_path: str = "/scratch/gsa/data/"
    add_bos: bool = False
    add_eos: bool = False
    workers: Optional[int] = None
    segments_per_chunk: int = 1
    glob_pattern: str = "*.jsonl"
    chunks: Optional[List[int]] = None
    code: Optional[str] = None
    output_file: Optional[str] = None
    config: Any = None

def parse_and_merge_config():
    cli_args = OmegaConf.from_cli()
    # file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    # del cli_args.config

    default_cfg = OmegaConf.structured(Args())
    cfg = OmegaConf.merge(default_cfg, cli_args)
    # cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    return cfg


def main():
    args = parse_and_merge_config()
    input_path = Path(args.input_path)

    # Determine optimal number of workers
    if args.workers is None:
        args.workers = min(cpu_count(), 64)
        logger.info(f"Auto-selected {args.workers} workers (CPUs: {cpu_count()})")
    else:
        args.workers = min(args.workers, 64)

    logger.info(f"Configuration:")
    logger.info(f" Workers: {args.workers}")
    # logger.info(f" Chunks to process: {len(args.chunks)}")
    logger.info(f" Segments per chunk: {args.segments_per_chunk}")
    # logger.info(f" Total work items: ~{len(args.chunks) * args.segments_per_chunk}")

    if args.tokenizer_name:
        tokenizer_args = TokenizerArgs(
            name=args.tokenizer_name,
            path=args.tokenizer_path,
            tokenizers=None,
        )
    else:
        tokenizer_args = args.tokenizer

    tokenizer = build_tokenizer(
        tokenizer_args.name, tokenizer_args.path, tokenizer_args.tokenizers
    )
    # exit(0)
    # Create all work items (chunk_id, segment_id, start_line, end_line)
    logger.info("Creating work items...")
    work_items = create_work_items(
        args.chunks, input_path, args.glob_pattern, args.segments_per_chunk
    )
    if not work_items:
        logger.error("No valid work items to process!")
        exit(1)

    logger.info(f"Created {len(work_items)} work items")
    # Create partial function with constant arguments
    worker_func = functools.partial(
        process_file_segment,
        tokenizer_args=tokenizer_args,
        args=args,
    )
    # Run in parallel
    logger.info(f"Starting processing with {args.workers} workers...")
    logger.info(
        f"Tokenizer: {args.tokenizer_name}, add_bos={args.add_bos}, add_eos={args.add_eos}"
    )
    start_time = time.time()
    with Pool(processes=args.workers) as pool:
        # Unpack work items into function arguments
        results = pool.starmap(worker_func, work_items)
        elapsed_time = time.time() - start_time
        # Aggregate results by chunk
        chunk_totals = {}
        chunk_bytes = {}
        token_count_per_source_aggregate = {}
        for (
            chunk_id,
            segment_id,
            token_count,
            total_bytes,
            token_count_per_source,
            file_path,
        ) in results:
            chunk_totals[chunk_id] = chunk_totals.get(chunk_id, 0) + token_count
            chunk_bytes[chunk_id] = chunk_bytes.get(chunk_id, 0) + total_bytes
            for source, count in token_count_per_source.items():
                token_count_per_source_aggregate[source] = (
                    token_count_per_source_aggregate.get(source, 0) + count
                )
    total_tokens = sum(chunk_totals.values())
    total_bytes = sum(chunk_bytes.values())
    # Summary statistics
    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total time: {elapsed_time:.1f}s")
    logger.info(f"Workers used: {args.workers}")
    logger.info(f"Segments per chunk: {args.segments_per_chunk}")
    logger.info(f"Total work items: {len(work_items)}")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Total data processed: {total_bytes / 1e9:.2f} GB")
    logger.info(f"Total tokens per source: {token_count_per_source_aggregate}")
    logger.info(f"Throughput: {total_tokens / elapsed_time:,.0f} tokens/sec")
    logger.info("\nToken counts per chunk:")

    for chunk_id in sorted(chunk_totals.keys(), key=lambda x: (x is None, x or 0)):
        logger.info(f" Chunk {str(chunk_id):>4}: {chunk_totals[chunk_id]:,} tokens")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info("=" * 80)
    # with open("/scratch/gsa/token_counts_per_source.yaml", "a") as f_out:
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, "a") as f_out:
            key = args.code if args.code else f"{args.tokenizer_path}_{args.input_path.name}"
            dct = {key: {"total_tokens": total_tokens, "total_bytes": total_bytes, "chunk_counts": chunk_totals, "token_count_per_source": token_count_per_source_aggregate,}}
            yaml.dump(dct, f_out)


if __name__ == "__main__":
    main()
