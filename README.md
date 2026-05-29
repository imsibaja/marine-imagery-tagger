# Marine Imagery Metadata Tagger

Automated LLM-assisted annotation pipeline for marine wildlife survey imagery metadata. Built as a proof-of-concept for NOAA SWFSC-style triage workflows that reduce manual analyst burden before downstream species distribution modeling or stock assessment inference.

## Overview

Field surveys from underwater gliders, aerial drones, and ROVs generate thousands of imagery records per deployment. Before images enter a computer vision pipeline, an analyst typically reviews each record to validate the species ID, flag data quality issues, and write a short field note — work that scales poorly at survey volumes.

This script automates that triage step. It sends each record's metadata to `claude-sonnet-4-6` via the Anthropic API and receives back a structured JSON annotation covering ecological classification, habitat inference, behavioral context, quality scoring, and a concise survey note. Results are written to an output CSV that can feed directly into a QA review queue or ML preprocessing step.

## Workflow

```
Input (CSV or image directory)
        │
        ▼
  load_csv() / load_folder()
        │  (list of metadata dicts)
        ▼
     process()
        │
        ├─ for each record:
        │       call_api()  →  Anthropic claude-sonnet-4-6
        │       ← JSON annotation
        │
        ▼
   write_output()  →  tagged_output.csv
```

### Prompt caching

The system prompt is marked with `cache_control: {"type": "ephemeral"}`. On batches of 50+ records, this reduces input token costs by ~90% after the first API call — the stable system prompt is read from Anthropic's cache on every subsequent request. The caching threshold is 2,048 tokens; embedding a species taxonomy table or survey SOP in the system prompt will push it past that threshold.

## Inputs

### CSV mode

A CSV with any subset of these columns (missing columns default to `"unknown"`):

| Column | Description |
|---|---|
| `image_id` | Unique identifier (filename stem or survey ID) |
| `species_observed` | Raw species label from image classifier or field notes |
| `location` | Named location or NMS area |
| `latitude` | Decimal degrees |
| `longitude` | Decimal degrees |
| `timestamp` | ISO 8601 or YYYYMMDD |
| `camera_type` | Platform: `AERIAL_DRONE`, `ROV`, `GLIDER`, `TRAIL_CAM`, `TOWED_CAM` |
| `depth_m` | Observation depth in meters |

See `example_input.csv` for 15 synthetic records spanning common California Current survey platforms.

### Directory mode

A folder of image files (`.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.cr2`, `.nef`). Metadata is parsed from filenames using the convention:

```
YYYYMMDD[_HHMMSS][_LOCATION][_CAMERA][_SPECIES].ext

Example: 20240315_143000_MBAY_AERIAL_DRONE_BlueWhale.jpg
```

## Outputs

Appended to `tagged_output.csv` (or `--output` path). New columns added by the LLM:

| Column | Description |
|---|---|
| `ecological_classification` | Validated species in taxonomic nomenclature |
| `habitat_type` | Inferred habitat (e.g., pelagic, kelp forest, benthic soft-sediment) |
| `behavior_notes` | Probable behavioral state or "insufficient metadata" |
| `quality_assessment` | Usability rating: Excellent / Good / Marginal / Poor |
| `quality_score` | Integer 1–5 for programmatic filtering |
| `survey_notes` | 1–2 sentence field database entry with QA flags |
| `confidence_level` | High / Medium / Low |
| `processed_at` | UTC timestamp of annotation |
| `model_used` | Model version string |

Records that fail after retries are written with `ecological_classification = "ERROR"` and `quality_score = 1` so the output CSV stays complete and downstream steps can filter rather than crash.

## Setup

**conda (recommended):**
```bash
conda env create -f environment.yml
conda activate marine-tagger
```

**pip:**
```bash
pip install -r requirements.txt
```

**API key:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# From a metadata CSV
python marine_tagger.py --input example_input.csv

# From a directory of image files
python marine_tagger.py --input /surveys/2024/mbay/ --output mbay_tagged.csv

# Debug logging to see token usage per record
python marine_tagger.py --input example_input.csv --log-level DEBUG
```

All runs append to the output CSV; re-runs on additional input batches accumulate into the same file without overwriting existing annotations.

## Scaling

| Scale | Approach |
|---|---|
| < 500 records | Run as-is; completes in minutes |
| 500–50,000 records | Add `--batch-size` chunking; submit via the [Anthropic Batches API](https://docs.anthropic.com/en/api/creating-message-batches) for 50% cost reduction and no rate-limit exposure |
| > 50,000 records | Parallelize with `concurrent.futures.ThreadPoolExecutor`; tune concurrency against your rate-limit tier |

The Batches API is the natural next step for full deployment: submit the entire survey's records as a batch job, poll for completion, and write results in one pass. The structured prompt and JSON output schema in this script are directly compatible with that pathway.

## Error handling

- **Rate limits**: reads the `retry-after` response header; falls back to exponential backoff with jitter
- **Server errors (5xx)**: exponential backoff, up to `MAX_RETRIES = 5` attempts
- **Malformed JSON**: logged and re-raised; record written as ERROR in output
- **Missing input metadata**: defaults to `"unknown"` / `"unidentified"` — the model handles sparse records gracefully and will lower its confidence rating accordingly
