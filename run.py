import os
import asyncio
import sys
import time
from logging_setup import setup_logging
from agentv8 import initialize_or_update
from collections import defaultdict

class Tee:
    """
    A simple tee that writes to multiple streams.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

async def process_all_patients(input_folder: str):
    setup_logging()
    """
    Process all .txt and .json files in the specified folder using initialize_or_update,
    skipping already-done files (logs with ✅ Done), and only retaining logs for successful runs.
    """
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    # include both .txt and .json files
    data_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.txt', '.json'))
    ]

    # group by patient_id
    files_by_id: dict[str, list[tuple[str,str]]] = defaultdict(list)
    for fname in data_files:
        stem = os.path.splitext(fname)[0]
        if "_" in stem:
            pid, ts = stem.split("_", 1)
        else:
            pid, ts = stem, ""
        files_by_id[pid].append((ts, fname))

    # prepare logs folder
    log_folder = os.path.join(input_folder, "logs")
    os.makedirs(log_folder, exist_ok=True)

    total_start = time.perf_counter()

    # process each patient in alphabetical order of ID
    for pid in sorted(files_by_id):
        sorted_files = sorted(files_by_id[pid], key=lambda x: x[0])
        for ts, data_file in sorted_files:
            file_path = os.path.join(input_folder, data_file)
            basename = os.path.splitext(data_file)[0]
            log_path = os.path.join(log_folder, f"{basename}.log")

            # skip if log exists and contains a success marker
            if os.path.isfile(log_path):
                try:
                    with open(log_path, 'r', encoding='utf-8') as lf:
                        content = lf.read()
                    if '✅ Done' in content.splitlines()[-1]:
                        print(f"⏭️ Skipping {data_file}, already done.")
                        continue
                except Exception:
                    # if unable to read, remove broken log and reprocess
                    os.remove(log_path)

            old_stdout, old_stderr = sys.stdout, sys.stderr
            success = False
            with open(log_path, "w", encoding="utf-8") as log_file:
                sys.stdout = Tee(old_stdout, log_file)
                sys.stderr = Tee(old_stderr, log_file)

                start = time.perf_counter()
                try:
                    print(f"\n📄 [{pid} @ {ts}] Processing file: {data_file}")
                    await initialize_or_update(file_path)
                    elapsed = time.perf_counter() - start
                    print(f"✅ Done {data_file} in {elapsed:.2f}s")
                    success = True
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    print(f"❌ Error on {data_file} after {elapsed:.2f}s: {e}")
                finally:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    sys.stdout, sys.stderr = old_stdout, old_stderr

            # remove log if unsuccessful
            if not success and os.path.isfile(log_path):
                os.remove(log_path)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n🏁 All files processed in {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_process_patients.py <input_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    asyncio.run(process_all_patients(input_folder))
