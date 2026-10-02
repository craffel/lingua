"""
Script to dump documents from log files in parallel by rank. Each process will handle one rank and write to a separate output file.
The script reads the log file line by line, extracts the source information for the specified rank, reads the corresponding document from the source file, and writes it to an output JSONL file.
"""
# # python apps/main/dump_docs.py --output_dir /scratch/gsa/data_recreation-dump --log_file llama_1B-data-recreation.log --record_source

import argparse
import ast
import json
import logging
import re
from multiprocessing import Process
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REGEX_PATTERN=r"^Rank\s(\d+)\s-\sChosen\sSource:\s([^\s|]+)\s\|\sSource\sState:\s(\{.*?\})"
FILE_REPLACE_PATTERN = ""
FILE_PATTERN_TO_REPLACE_WIITH = ""

def process_rank(rank, output_dir, log_file, record_source, num_chunks, buffer_size=10000):
    """Process all lines for a specific rank"""
    output_path = Path(output_dir) / f"train_data.chunk.{rank:02d}.jsonl"
    
    # Clear the output file
    with open(output_path, "w") as f_out:
        pass
    
    rank_buffer = []
    lines_written = 0
    lines_processed = 0
    
    with open(log_file, "r") as log_f:
        for line in log_f:
            lines_processed += 1
            match = re.match(REGEX_PATTERN, line)
            if match:
                matched_rank = int(match.group(1))
                
                # Skip if not our rank
                if matched_rank != rank:
                    continue
                
                source = match.group(2)
                try:
                    source_state = ast.literal_eval(match.group(3))
                except Exception as e:
                    logger.error(f"Rank {rank}: Error parsing source state: {e}")
                    continue
                
                file_path = source_state['file_path']
                file_path = file_path.replace(FILE_REPLACE_PATTERN, FILE_PATTERN_TO_REPLACE_WIITH)
                position = source_state['position']
                
                try:
                    with open(file_path, "r") as f_in:
                        f_in.seek(position)
                        doc = f_in.readline()
                        doc = json.loads(doc)["text"]
                        data = {"text": doc}
                        if record_source:
                            data["source"] = source
                            data["position"] = position
                        rank_buffer.append(data)
                except Exception as e:
                    logger.error(f"Rank {rank}: Error reading file {file_path} at position {position}: {e}")
                    continue
                
                # Write buffer periodically
                if len(rank_buffer) >= buffer_size:
                    with open(output_path, "a") as f_out:
                        for record in rank_buffer:
                            f_out.write(json.dumps(record) + "\n")
                            lines_written += 1
                    logger.info(f"Rank {rank}: Wrote {lines_written} documents so far...")
                    rank_buffer = []
            
            # Progress update
            if lines_processed % 100000 == 0:
                logger.info(f"Rank {rank}: Processed {lines_processed} log lines...")
    
    # Write remaining buffer
    if rank_buffer:
        with open(output_path, "a") as f_out:
            for record in rank_buffer:
                f_out.write(json.dumps(record) + "\n")
                lines_written += 1
    
    logger.info(f"Rank {rank}: COMPLETE - Total documents written: {lines_written}")

def dump_training_data_parallel(output_dir, log_file, record_source, num_chunks=8, buffer_size=10000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting {num_chunks} parallel processes...")
    
    # Spawn one process per rank
    processes = []
    for rank in range(num_chunks):
        p = Process(
            target=process_rank,
            args=(rank, output_dir, log_file, record_source, num_chunks, buffer_size)
        )
        p.start()
        processes.append(p)
        logger.info(f"Started process for rank {rank}")
    
    # Wait for all processes to complete
    for rank, p in enumerate(processes):
        p.join()
        logger.info(f"Rank {rank} process completed")
    
    logger.info("All processes completed!")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to dump training data")
    parser.add_argument("--log_file", type=str, default="dump_docs.log", help="Log file name")
    parser.add_argument("--record_source", action="store_true", help="Whether to record source information")
    parser.add_argument("--num_chunks", type=int, default=8, help="Number of chunks/ranks")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Buffer size before writing")
    args = parser.parse_args()
    
    dump_training_data_parallel(
        args.output_dir, 
        args.log_file, 
        args.record_source, 
        args.num_chunks,
        args.buffer_size
    )