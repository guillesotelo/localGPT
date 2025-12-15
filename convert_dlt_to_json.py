from pydlt import DltFileReader, DltFileWriter
import json
import re
import logging
from copy import deepcopy
import pickle

input_file = "input/DLT_DIR.dlt"
output_file = "input/DLT_DIR.json"

def clean_text(text):
    """
    Normalize text safely whether it's bytes or str.
    Always returns a str.
    """
    if text is None:
        return ""

    if isinstance(text, bytes):
        # Safe decode
        return text.decode("utf-8", errors="replace")

    if isinstance(text, str):
        # Normalize
        return text.encode("utf-8", errors="replace").decode("utf-8")

    # Unexpected type → force string
    return str(text)


def parse_logs(input_file, output_file):
    print("Starting conversion...")
    logs = []
    entries = 0
    with DltFileReader(input_file) as reader:
        for index, msg in enumerate(reader):
            entries += index
            # msg is a DltMessage
            verbose_str = str(msg)
            verbose_str = clean_text(verbose_str)  # clean invalid characters

            # Example: parse the timestamp at the start (YYYY/MM/DD HH:MM:SS.ssssss)
            ts_match = re.match(r'^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d+)', verbose_str)
            timestamp = ts_match.group(1) if ts_match else None
            
            if " verbose 1 " in verbose_str:
                payload = verbose_str.split(" verbose 1 ", 1)[1].strip()
            else:
                payload = None  # fallback if "log" not found

            payload = clean_text(payload)  # clean payload too

            log_entry = {
                "timestamp": timestamp,
                "raw_verbose": verbose_str,
                "payload": payload,
            }

            logs.append(log_entry)
            
    # Save as JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)  # keep UTF-8 chars

    print(f"Done converting {entries} entries.")
    return entries


def parse_original_logs(input_file, output_file):
    try:
        logging.info(f"Parsing DLT to JSON...")
        logs = []
        packets = []

        with DltFileReader(input_file) as reader:
            for msg in reader:
                packets.append(deepcopy(msg))
                verbose_str = clean_text(str(msg))

                ts_match = re.match(r'^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d+)', verbose_str)
                timestamp = ts_match.group(1) if ts_match else None

                payload = (
                    verbose_str.split(" verbose 1 ", 1)[1].strip()
                    if " verbose 1 " in verbose_str else None
                )

                logs.append({
                    "timestamp": timestamp,
                    "raw_verbose": verbose_str,
                    "payload": clean_text(payload)
                })
            # Save raw packets

        # Write log JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)


        logging.info(f"Done converting {len(logs)} entries.")
    except Exception as e:
        logging.error("Error in parse_original_logs: %s", str(e))



if __name__ == '__main__':
    parse_logs(input_file, output_file)
