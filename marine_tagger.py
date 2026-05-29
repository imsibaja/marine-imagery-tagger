#!/usr/bin/env python3
"""
marine_tagger.py — Automated Marine Imagery Metadata Tagger

Uses the Anthropic API (claude-sonnet-4-6) to classify, quality-assess, and
annotate marine wildlife imagery metadata at scale. Accepts a CSV of pre-extracted
metadata or a directory of image files and appends LLM-generated ecological
annotations to a structured output CSV.
"""

import anthropic
import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("marine_tagger.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 5
BASE_RETRY_DELAY = 1.0  # seconds

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".cr2", ".nef"}

# Output column order
OUTPUT_COLUMNS = [
    "image_id", "species_observed", "location", "latitude", "longitude",
    "timestamp", "camera_type", "depth_m",
    "ecological_classification", "habitat_type", "behavior_notes",
    "quality_assessment", "quality_score", "survey_notes",
    "confidence_level", "processed_at", "model_used",
]

# ── Prompt Construction ───────────────────────────────────────────────────────

# Substantial system prompt — embed domain reference so prompt caching amortizes
# across every record in a batch (cache_control marks the prefix cacheable).
# Sonnet 4.6 caches prefixes ≥ 2 048 tokens; expand with species field guides,
# survey SOPs, or taxonomy tables to reach that threshold and cut input costs ~90%.
SYSTEM_PROMPT = """\
You are an expert marine wildlife ecologist and field survey analyst supporting \
NOAA's Southwest Fisheries Science Center (SWFSC) Ecosystem Science Division. \
You specialize in interpreting metadata from large-scale automated imagery surveys \
conducted by underwater gliders, aerial drones, and remotely-operated vehicles (ROVs). \
Your outputs feed directly into species distribution models and stock assessment \
pipelines, so scientific precision, consistency, and calibrated uncertainty are \
critical.

## Your responsibilities for each record

1. ECOLOGICAL CLASSIFICATION — Validate or correct the provided species name \
using all available metadata (location, depth, camera platform, season). Apply \
standard taxonomic nomenclature (genus species). If multiple species are plausible, \
list the most likely one and note alternatives in survey_notes.

2. HABITAT CHARACTERIZATION — Infer the most likely habitat type from depth, \
latitude/longitude, platform type, and season. Common categories: open water / \
pelagic, kelp forest, rocky reef, benthic soft-sediment, sandy shelf, intertidal, \
estuarine.

3. BEHAVIORAL CONTEXT — If the metadata supports it, note probable behavioral \
state (foraging, resting, transiting, socializing, breeding, unknown). Use \
"insufficient metadata" when not inferrable.

4. DATA QUALITY ASSESSMENT — Rate usability for downstream ML model inference \
on a scale of 1–5:
   5 = Excellent: precise geolocation, high-quality platform, clear species ID
   4 = Good: minor gaps but suitable for most analyses
   3 = Marginal: usable with caveats; note limitations
   2 = Poor: significant issues; use with caution
   1 = Unusable: missing critical metadata or implausible observation

5. SURVEY NOTES — Write 1–2 sentences as a concise scientific record entry \
suitable for a NOAA field database. Include any data quality flags.

6. CONFIDENCE LEVEL — Assign High / Medium / Low confidence to your overall \
annotation, considering data completeness and species identification certainty.

## Platform context

- AERIAL_DRONE / AERIAL: surface-visible species only; depth = 0
- GLIDER / AUV: subsurface observations; depth is reliable
- ROV / REMOTELY_OPERATED: precise depth; high spatial resolution
- TRAIL_CAM: shoreline / haul-out / intertidal species
- TOWED_CAM: depth = tow depth; variable image quality

## Critical rules
- Return ONLY a valid JSON object — no prose, no markdown fences, no commentary.
- All string values must be populated; use "unknown" or "insufficient metadata" \
  rather than empty strings or null.
- quality_score must be an integer 1–5.
- confidence_level must be exactly "High", "Medium", or "Low".
- quality_assessment must be exactly "Excellent", "Good", "Marginal", or "Poor".\
"""


def build_user_prompt(record: dict) -> str:
    """Format a single metadata record into a structured analysis request."""
    return f"""\
Analyze the following marine imagery metadata record and return a JSON object \
with your ecological annotations.

METADATA:
  image_id:      {record.get('image_id', 'N/A')}
  species:       {record.get('species_observed', 'unidentified')}
  location:      {record.get('location', 'N/A')}
  latitude:      {record.get('latitude', 'N/A')}
  longitude:     {record.get('longitude', 'N/A')}
  timestamp:     {record.get('timestamp', 'N/A')}
  camera_type:   {record.get('camera_type', 'N/A')}
  depth_m:       {record.get('depth_m', 'N/A')}

Return exactly this JSON structure (no other text):
{{
  "ecological_classification": "<validated species name>",
  "habitat_type": "<habitat category>",
  "behavior_notes": "<behavioral context or 'insufficient metadata'>",
  "quality_assessment": "<Excellent|Good|Marginal|Poor>",
  "quality_score": <integer 1–5>,
  "survey_notes": "<1–2 sentence scientific summary>",
  "confidence_level": "<High|Medium|Low>"
}}\
"""


# ── API Layer ─────────────────────────────────────────────────────────────────

def call_api(client: anthropic.Anthropic, record: dict) -> dict:
    """
    Call the Anthropic API for a single record with exponential-backoff retry.

    Caches the stable system prompt across all records in a batch.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=512,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        # Mark system prompt cacheable — on batches of 50+ records
                        # this reduces input token costs by ~90% after the first call.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {"role": "user", "content": build_user_prompt(record)}
                ],
            )

            raw = response.content[0].text.strip()

            # Strip accidental markdown code fences
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(
                    l for l in lines
                    if not l.strip().startswith("```")
                ).strip()

            annotations = json.loads(raw)

            log.debug(
                "Record %s — input_tokens=%d  cached=%d",
                record.get("image_id"),
                response.usage.input_tokens,
                response.usage.cache_read_input_tokens or 0,
            )
            return annotations

        except anthropic.RateLimitError as exc:
            retry_after = float(
                exc.response.headers.get("retry-after", BASE_RETRY_DELAY * (2 ** attempt))
            )
            delay = retry_after + random.uniform(0, 1)
            log.warning(
                "Rate limited (attempt %d/%d). Waiting %.1fs …",
                attempt + 1, MAX_RETRIES, delay,
            )
            time.sleep(delay)

        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                delay = BASE_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                log.warning(
                    "Server error %d (attempt %d/%d). Waiting %.1fs …",
                    exc.status_code, attempt + 1, MAX_RETRIES, delay,
                )
                time.sleep(delay)
            else:
                log.error("Non-retryable API error (%d): %s", exc.status_code, exc.message)
                raise

        except (json.JSONDecodeError, IndexError) as exc:
            log.error("Could not parse API response for %s: %s", record.get("image_id"), exc)
            raise

    raise RuntimeError(
        f"Exhausted {MAX_RETRIES} retries for record '{record.get('image_id')}'"
    )


# ── Input Loaders ─────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    """Load metadata records from a CSV file."""
    df = pd.read_csv(path, dtype=str).fillna("")
    records = df.to_dict("records")
    log.info("Loaded %d records from %s", len(records), path)
    return records


def load_folder(path: Path) -> list[dict]:
    """
    Build metadata records from image filenames in a directory.

    Expected filename convention (all parts after date are optional):
      YYYYMMDD[_HHMMSS][_LOCATION][_CAMERA][_SPECIES].ext

    Example: 20240315_143000_MBAY_AERIAL_DRONE_BluWhale.jpg
    """
    records = []
    for fp in sorted(path.iterdir()):
        if fp.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parts = fp.stem.split("_")
        # Extract parts by position where available
        image_id = fp.stem
        timestamp = parts[0] if len(parts) > 0 else ""
        location = parts[2] if len(parts) > 2 else "unknown"
        camera = "_".join(parts[3:5]) if len(parts) > 4 else (parts[3] if len(parts) > 3 else "unknown")
        species = parts[5] if len(parts) > 5 else "unidentified"

        records.append({
            "image_id": image_id,
            "species_observed": species,
            "location": location,
            "latitude": "",
            "longitude": "",
            "timestamp": timestamp,
            "camera_type": camera,
            "depth_m": "",
        })

    log.info("Discovered %d image files in %s", len(records), path)
    return records


# ── Output Writer ─────────────────────────────────────────────────────────────

def write_output(rows: list[dict], output_path: Path) -> None:
    """Append annotated rows to the output CSV, creating headers on first write."""
    df = pd.DataFrame(rows)
    # Ensure all expected columns are present (fill missing with empty string)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[OUTPUT_COLUMNS]

    write_header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=write_header, index=False)
    log.info("Wrote %d rows → %s", len(rows), output_path)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def process(
    client: anthropic.Anthropic,
    records: list[dict],
    output_path: Path,
) -> None:
    """Annotate every record through the LLM and stream results to CSV."""
    results = []
    n_ok = 0

    for i, record in enumerate(records, 1):
        image_id = record.get("image_id", f"row_{i}")
        log.info("[%d/%d] Processing %s …", i, len(records), image_id)

        try:
            annotations = call_api(client, record)
            results.append({
                **record,
                **annotations,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "model_used": MODEL,
            })
            n_ok += 1
        except Exception as exc:
            log.error("Failed to process %s: %s", image_id, exc)
            results.append({
                **record,
                "ecological_classification": "ERROR",
                "habitat_type": "",
                "behavior_notes": "",
                "quality_assessment": "Poor",
                "quality_score": 1,
                "survey_notes": f"Processing error: {exc}",
                "confidence_level": "Low",
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "model_used": MODEL,
            })

    write_output(results, output_path)
    log.info(
        "Complete — %d/%d records annotated successfully.",
        n_ok, len(records),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Marine Imagery Metadata Tagger (Anthropic API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python marine_tagger.py --input example_input.csv
  python marine_tagger.py --input /surveys/2024/images/ --output tagged_output.csv
  python marine_tagger.py --input data.csv --output results.csv --log-level DEBUG
        """,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV or directory of image files",
    )
    parser.add_argument(
        "--output", default="tagged_output.csv",
        help="Path to output CSV (appended if it already exists; default: tagged_output.csv)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running: export ANTHROPIC_API_KEY=sk-ant-..."
        )
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        records = load_csv(input_path)
    elif input_path.is_dir():
        records = load_folder(input_path)
    else:
        log.error("--input must be a .csv file or a directory of images.")
        sys.exit(1)

    if not records:
        log.warning("No records found — nothing to process.")
        sys.exit(0)

    process(client, records, Path(args.output))


if __name__ == "__main__":
    main()
