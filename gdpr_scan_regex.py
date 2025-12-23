#!/usr/bin/env python3
import json
import re
import argparse
import shutil
import multiprocessing as mp
from tqdm import tqdm
from collections import Counter
from copy import deepcopy
from pydlt import DltFileReader, DltFileWriter
from pydlt.message import DltMessage
from pydlt.payload import VerbosePayload
from constants import (
    SEVERITY_MAP,
    BIOMETRIC_WORDS,
    IN_CABIN_CAMERA,
    CRIMINAL_INDICATORS,
    HTML_TEMPLATE
)
import logging

# -------------------------
# Regex Patterns
# -------------------------
REGEX_PATTERNS = {
    "VIN": re.compile(r"\b(?=[A-HJ-NPR-Z0-9]{17}\b)(?=.*[A-HJ-NPR-Z])(?=.*\d)[A-HJ-NPR-Z0-9]{17}\b"),
    "MAC_ADDRESS": re.compile(r"\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
    "UUID": re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"),
    "IP_ADDRESS": re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b"),
    # "DEVICE_ID": re.compile(r"\b[A-Za-z0-9_]{10,}\b"),
    "NAME": re.compile(r"\b([A-ZÅÄÖ][a-zà-öø-ÿ]+(?:[-'][A-ZÅÄÖ][a-zà-öø-ÿ]+)?)\s+([A-ZÅÄÖ][a-zà-öø-ÿ]+(?:[-'][A-ZÅÄÖ][a-zà-öø-ÿ]+)?)\b"),
    "EMAIL_ADDRESS": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE_NUMBER": re.compile(
          r"""
        (?<!\d)                    # not preceded by digit
        (?:\+?\d{1,3}[\s-]?)?      # optional country code
        (?:\(?\d{2,4}\)?[\s-]?)    # optional area code
        \d{3}[\s-]\d{4}            # main number
        (?!\d)                      # not followed by digit
        """, re.VERBOSE),
    "GPS_COORDINATES": re.compile(
        r"""
        (?:
            # Explicit latitude + longitude fields (any order, same line)
            (?:
                (?<![a-zA-Z])
                (?:lat|latitude)\s*[:=]\s*([-+]?\d{1,2}\.\d{4,})
                .*?
                (?:lon|longitude)\s*[:=]\s*([-+]?\d{1,3}\.\d{4,})
            )
            |
            # GNSS / automotive structured location payloads
            (?:
                \b(setGNSS|GNSSRawLocation|gnss_ts|rawLocationData)\b
            )
        )
        """,
        re.IGNORECASE | re.VERBOSE
    ),
    "CREDIT_CARD": re.compile( r"""
        (?:  
            (?<![A-Fa-f0-9])
            \d{16}                
            (?![A-Fa-f0-9])
        )
        |
        (?:\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4})
        """,
    re.VERBOSE),
    "SW_SECURITY_NUMBER": re.compile(
        r"\b(19[0-9]{2}|20[0-2][0-9])"
        r"(0[1-9]|1[0-2])"
        r"(0[1-9]|[12][0-9]|3[01]|6[1-9]|[78][0-9]|9[0-1])"
        r"-?\d{4}\b"
        ),
    
    "ALCOHOL": re.compile(r"(alcohol|\bbac\b|blood\s*alcohol|breath(?:alyzer)?|ethanol|alko(lock)?)", re.I),
    "BIOMETRIC": re.compile(r"(heart\s*rate|fatigue|weight|stress\s*level|drowsiness|sleepiness)", re.I),
    "DMS_OMS": re.compile(r"\b(dms|driver\s*monitoring|driver\s*state|oms|occupant\s*monitoring)\b", re.I),
    "SPEED": re.compile(r"(speed|velocity)\s*[:=]?\s*\d{2,3}", re.I),
    "SAFETY_EVENTS": re.compile(
        r"(\bcrash\b|collision|impact|airbag|hard\s*brake|emergency\s*brake|seatbelt\s*(warning|unbuckled))",
        re.I),
    "VIOLATIONS": re.compile(
        r"(speeding|failure\s*to\s*stop|illegal|violation|offence|ticket)", re.I),
    "CAMERA": re.compile(r"(camera|image\s*analysis|object\s*detection|in-cabin|\bface\b|\bgaze\b|eye\s*(tracking|closure))", re.I),
    "BEHAVIORAL_STATE": re.compile(
        r"(aggressive\s*driving|harsh\s*(steer|brake|accel)|swerving)", re.I),
}


# -------------------------
# Match entry-severity
# -------------------------
def severity_for_entity(entity):
    return SEVERITY_MAP.get(entity, "low")

# -------------------------
# Clean log text
# -------------------------
def clean_text(text):
    """
    Normalize text safely (bytes/str), remove long hex blobs.
    Always returns a str.
    """
    HEX_BLOB = re.compile(r"\b[0-9A-Fa-f]{20,}\b")

    if text is None:
        return ""

    # If it's bytes → decode only
    if isinstance(text, bytes):
        t = text.decode("utf-8", errors="replace")

    # If it's str → normalize encode/decode
    elif isinstance(text, str):
        t = text.encode("utf-8", errors="replace").decode("utf-8")

    # Other types → stringify safely
    else:
        t = str(text)

    # Remove long hex/noise blocks
    t = HEX_BLOB.sub(" ", t)

    return t



# -------------------------
# Luhn check for Credit Cards
# -------------------------
def luhn_check(cc_num):
     # Remove spaces/dashes
    cc = re.sub(r"\D", "", cc_num)
    # Reject if all digits are the same
    if len(set(cc)) == 1:
        return False
    # Luhn check
    total = 0
    total = 0
    reverse_digits = cc_num[::-1]
    for i, d in enumerate(reverse_digits):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# -------------------------
# Reject false positives on Name
# -------------------------
WORD_DB = set()
with open("data/english_words.txt") as f:
    for line in f:
        WORD_DB.add(line.strip().lower())

TECH_WORDS_STR = "hib"
WORD_DB |= set(word.lower() for word in TECH_WORDS_STR.split())


# -------------------------
# Check combos like GPS and Speed which can show a criminal act
# e.g., SPEED + LOCATION → criminal implication (speeding)
#      SPEED + TIME + POSITION → event reconstruction
# -------------------------
def is_sensitive_combo(entity_type, text):
    t = text.lower()
    
    # Biometric
    if entity_type not in ['ALCOHOL', 'BIOMETRIC']:
        if any(word in t for word in BIOMETRIC_WORDS):
            return True

    # In-cabin camera
    if entity_type not in ['CAMERA', 'DMS_OMS']:
        if any(cam in t for cam in IN_CABIN_CAMERA):
            return True

    # Criminal/incriminating
    if entity_type not in ['SPEED', 'VIOLATIONS', 'BEHAVIORAL_STATE']:
        if any(crim in t for crim in CRIMINAL_INDICATORS):
            return True

    # GPS & GNNS
    if entity_type not in ['GPS_COORDINATES']:
        if re.search(
                r"\b("
                r"lat|lon|latitude|longitude|"
                r"gnss|setgnss|gnss_ts|rawlocation|gps"
                r")\b",
                t,
                re.IGNORECASE
            ):
            return True

    # --- SPEED COMBO ---
    if entity_type == "SPEED":
        has_speed = re.search(r"(?<![a-zA-Z])(speed|spd|vehicle_speed)(?![a-zA-Z])\s*[:=]?\s*\d{1,3}", t)
        has_gps = re.search(r"(?<![a-zA-Z])(lat|lon|latitude|longitude)(?![a-zA-Z])", t)
        has_zone = re.search(r"(?<![a-zA-Z])(road|zone|limit|speed_limit)(?![a-zA-Z])", t)

        if has_speed and (has_gps or has_zone):
            return True

    return False


# -------------------------
# Check false positives for Person Names and their combination with other data
# -------------------------
def is_false_positive_name(text, match):
    s, e = match.start(), match.end()
    name = match.group()
    # reject if tokens contain digits or are too long/short
    tokens = [t for t in re.split(r"\s+", name) if t]
    # reject if one token is English/technical word
    if any(t.lower() in WORD_DB for t in tokens):
        return True
    if any(re.search(r"\d", t) for t in tokens): 
        return True
    if any(len(t) < 2 or len(t) > 30 for t in tokens): 
        return True
    # reject if immediately after colon or inside brackets
    before = text[max(0, s-3):s]
    after = text[e:e+20].lower()
    if before.endswith(':') or before.endswith(' -') or '[' in text[max(0,s-10):s]:
        return True
    # reject if any token is clearly technical (camel/underscore/ALLCAPS)
    if any(re.search(r"[A-Z]{2,}", t) or "_" in t for t in tokens):
        return True
    return False


# -------------------------
# Check log severity
# -------------------------
def is_severe(entry):
    return severity_for_entity(entry) == 'high' or severity_for_entity(entry) == 'medium'


# -------------------------
# Scan a single log entry
# -------------------------
def scan_entry(entry, prev_findings=None):
    flagged = []
    text = clean_text(entry.get("raw_verbose", ""))
    prev_text = clean_text(prev_findings["raw_verbose"]) if prev_findings else ""

    for entity, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            
            if entity == "NAME" and is_false_positive_name(text, match):
                continue
            if entity == "CREDIT_CARD" and not luhn_check(match.group()):
                continue
            if entity == "GPS_COORDINATES":
                # If regex captured latitude/longitude, validate them
                if match.lastindex and match.lastindex >= 2:
                    try:
                        lat = float(match.group(1))
                        lon = float(match.group(2))
                        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                            continue
                    except Exception:
                        continue

            flagged.append({
                "entity_type": entity,
                "start": match.start(),
                "end": match.end(),
                "value": match.group(),
                "severity": severity_for_entity(entity)
            })

    # NEIGHBOR COMBO CHECK
    if prev_findings:
        combined_text = f"{prev_text} {text}".lower()
        found_types = {f["entity_type"] for f in flagged}

        prev_findings_list = prev_findings.get("findings", [])

        # Base entity types seen in previous log
        prev_base = {
            f["entity_type"].replace("SENSITIVE_COMBO_", "")
            for f in prev_findings_list
        }

        # Combo alerts already emitted in previous log
        prev_combos = {
            f["entity_type"] for f in prev_findings_list
        }

        for etype in found_types:
            base = etype.replace("SENSITIVE_COMBO_", "")
            combo_tag = f"SENSITIVE_COMBO_{base}"

            # suppression logic:
            prev_had_base = base in prev_base          # prev had alcohol
            prev_had_combo = combo_tag in prev_combos  # prev already emitted combo

            if (
                not prev_had_base             # prev log has entity
                and not prev_had_combo    # don't re-trigger the combo
                and is_severe(base)
                and is_sensitive_combo(base, combined_text)  # combo valid
            ):
                flagged.append({
                    "entity_type": combo_tag,
                    "value": (
                        "Sensitive data detected in combination of previous and current log:\n"
                        "<ul>"
                            f"<li><strong>Previous:</strong> {prev_text}</li>\n"
                            f"<li><strong>Current:</strong> {text}</li>"
                        "</ul>"
                    ),
                    "severity": "high"
                })


    if flagged:
        entry["findings"] = flagged
        return entry
    return None


# -------------------------
# Multiprocessing scanner
# -------------------------
def process_logs(entries, workers=mp.cpu_count()):
    try:
        sev_order = {"low": 0, "medium": 1, "high": 2}
        flagged_total = []

        prev_entry = None

        for entry in tqdm(entries, desc="Scanning logs"):
            # Run heavy regex in parallel
            flagged_entry = scan_entry(entry, prev_findings=prev_entry)
            if flagged_entry:
                flagged_total.append(flagged_entry)

            # Save current entry as previous for next iteration
            prev_entry = flagged_entry if flagged_entry else {"raw_verbose": entry.get("raw_verbose", "")}

        # Sort by severity
        def entry_severity(entry):
            return max(sev_order[f["severity"]] for f in entry["findings"])

        flagged_total.sort(key=entry_severity, reverse=True)
        return flagged_total
    except Exception as e:
        logging.error("Error in process_logs: %s", str(e))

# -------------------------
# HTML page generator with entries and their flags
# -------------------------
def generate_html_report(flagged_entries, output_file):
    # --- Compute summary counts ---
    summary_counter = Counter()
    has_high = False

    for e in flagged_entries:
        for f in e["findings"]:
            key = (f["entity_type"], f["severity"])
            summary_counter[key] += 1
            if f["severity"] == "high":
                has_high = True

    # GDPR compliance logic
    summary_class = "bad" if has_high else "good"
    gdpr_message = "NO. (High-severity personal data detected)" if has_high else "YES (No high-severity findings)"

    # Build summary rows
    sev_css = {"high": "sev-label-high", "medium": "sev-label-medium", "low": "sev-label-low"}

    summary_rows = []
    for (etype, sev), count in sorted(summary_counter.items(), key=lambda x: (-{"high":2,"medium":1,"low":0}[x[0][1]], x[0][0])):
        summary_rows.append(
            f"<tr>"
            f"<td class='entity-type'>{etype}</td>"
            f"<td class='{sev_css[sev]}'>{sev.upper()}</td>"
            f"<td>{count}</td>"
            f"</tr>"
        )
    summary_html = "\n".join(summary_rows)

    # --- Build entry blocks ---
    blocks = []
    sev_order = {"low":0, "medium":1, "high":2}

    for e in flagged_entries:
        highest_sev = max([f["severity"] for f in e["findings"]], key=lambda x: sev_order[x])
        css_class = f"sev-{highest_sev}"
        findings_text = "\n".join([f"<span class='entry-finding-sev-{f['severity']}'>{f['entity_type']}</span> ({f['severity'].upper()}): {f['value']}" for f in e["findings"]])

        block = f"""
        <div class="entry {css_class}">
            <h3 style="margin-top: 0;">{e['timestamp']} -> Severity: {highest_sev.upper()}</h3>
            <pre>{e.get('payload','')}</pre>
            <pre>{findings_text}</pre>
        </div>
        """
        blocks.append(block)

    html = HTML_TEMPLATE.format(
        count=len(flagged_entries),
        entries="\n".join(blocks),
        summary_rows=summary_html,
        summary_class=summary_class,
        gdpr_message=gdpr_message
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
        
    # Copy UI files to output
    file_name = output_file.split('/')[-1]
    output_folder = output_file.split(file_name)[0]
    shutil.copyfile('ui/styles.css', f"{output_folder}/styles.css")
    shutil.copyfile('ui/main.js', f"{output_folder}/main.js")
    


# -------------------------
# Safely replace DLTMessage payloads
# -------------------------
replace_logged = False
def replace_verbose_payload(msg):
    global replace_logged
    new_msg = deepcopy(msg)
    payload = getattr(new_msg, "payload", None)

    if not isinstance(payload, VerbosePayload):
        raise ValueError(f"Unsupported payload type: {type(payload)}")

    args = getattr(payload, "arguments", None)
    if not args:
        raise ValueError("VerbosePayload has no arguments")

    arg = args[0]  # first argument contains the verbose text

    original = getattr(arg, "data", None)
    if original is None:
        raise ValueError("VerbosePayload argument has no `.data` attribute")

    # Log once
    if not replace_logged:
        logging.info(f"********* ARG BEFORE MASKING: {original}")
        replace_logged = True

    masked = "*" * len(original)

    # ----------------------------------------------------------------------
    # 1. TRY DIRECT ASSIGNMENT
    # ----------------------------------------------------------------------
    try:
        arg.data = masked
        # Check if the assignment actually worked
        if getattr(arg, "data", None) == masked:
            return new_msg
    except Exception:
        pass

    # ----------------------------------------------------------------------
    # 2. HANDLE IMMUTABLE STRUCT (namedtuple, frozen dataclass, etc.)
    # ----------------------------------------------------------------------
    # Rebuild argument with masked data
    try:
        # For classes with __dict__ or slots-based instantiation
        new_arg = type(arg)(**{
            field: (masked if field == "data" else getattr(arg, field))
            for field in arg.__dataclass_fields__  # dataclass case
        })
    except Exception:
        try:
            # Generic namedtuple case
            if hasattr(arg, "_fields"):
                new_values = []
                for f in arg._fields:
                    new_values.append(masked if f == "data" else getattr(arg, f))
                new_arg = type(arg)(*new_values)
            else:
                raise
        except Exception:
            # FINAL FALLBACK – raw object replacement
            new_arg = deepcopy(arg)
            object.__setattr__(new_arg, "data", masked)

    # Replace the argument in the list
    args[0] = new_arg

    return new_msg



# -------------------------
# Mask DLT logs based on flagged timestamps
# -------------------------
def filter_and_mask_dlt(input_file, output_file, flagged_timestamps=set()):
    try:
        kept_count = 0
        masked_count = 0

        with DltFileReader(input_file) as reader:
            # IMPORTANT: don't let writer close the BytesIO: use explicit writer
            writer = DltFileWriter(output_file)

            for msg in reader:
                verbose_str = str(msg)
                verbose_str = clean_text(verbose_str)

                ts_match = re.match(r'^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d+)', verbose_str)
                timestamp = ts_match.group(1) if ts_match else None

                if timestamp in flagged_timestamps:
                    # Perform masking
                    new_msg = replace_verbose_payload(msg)
                    masked_count += 1
                else:
                    kept_count += 1
                    new_msg = msg
                
                try:
                    writer.write_message(new_msg)
                except Exception:
                    logging.exception("filter_and_mask_dlt: writer.write_message failed")
                    writer.write_message(msg)
                    continue

            try:
                # explicitly close writer but do not close the stream object (pydlt handles its own resources)
                writer.close()
            except Exception:
                logging.exception("filter_and_mask_dlt: writer.close() raised")

        logging.info("Filtered & masked DLT: %d kept, %d masked.", kept_count, masked_count)
        return True

    except Exception:
        logging.exception("Error in filter_and_mask_dlt")
        return False


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Regex-based GDPR scanner for DLT logs")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file (flagged)")
    parser.add_argument("--report", default="output/gdpr_report.html", help="Optional HTML report file")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        logs = json.load(f)

    flagged = process_logs(logs, workers=args.workers)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(flagged, f, indent=2, ensure_ascii=False)

    generate_html_report(flagged, args.report)
    print(f"Done. Flagged {len(flagged)} entries.")
    print(f"JSON: {args.output}")
    print(f"HTML report: {args.report}")

if __name__ == "__main__":
    main()
