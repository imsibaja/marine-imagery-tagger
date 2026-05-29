# Building an LLM-Assisted Triage Pipeline for Marine Survey Imagery

Modern marine wildlife surveys generate data faster than analysts can review it. A single glider deployment across the California Current can produce thousands of imagery records over two weeks — each one needing a validated species ID, a habitat tag, a data quality flag, and a field note before it can enter a species distribution model or stock assessment pipeline.

The manual triage step is the bottleneck. This post walks through a Python pipeline I built that automates it using the Anthropic API, turning raw survey metadata into structured ecological annotations at scale.

---

## The Problem

Survey platforms — aerial drones, underwater gliders, ROVs, trail cameras — produce metadata records that look something like this:

| Field | Value |
|---|---|
| `image_id` | `20240318_GLDR_007` |
| `species_observed` | `Common Dolphin` |
| `location` | `Central CA Coast` |
| `latitude / longitude` | `36.49°N, 122.91°W` |
| `timestamp` | `2024-03-18T15:03:00` |
| `camera_type` | `GLIDER` |
| `depth_m` | `22` |

Before this record can feed into a downstream model, an analyst needs to:

1. Validate the species ID against location, depth, and season
2. Flag whether *Delphinus delphis* or *Delphinus capensis* is more likely given the offshore position
3. Characterize the habitat and probable behavioral state
4. Score the record's usability for ML inference (image quality, metadata completeness)
5. Write a concise field note suitable for a NOAA database

Multiply that by 50,000 records per survey season and the math doesn't work. LLMs are a natural fit — they have broad ecological knowledge, can reason over sparse metadata, and return calibrated uncertainty.

---

## Architecture

The pipeline is intentionally simple: one Python script, one input format, one output CSV.

```
Input CSV (or image directory)
        │
        ▼
  load records into dicts
        │
        ▼
  for each record:
    POST /v1/messages  →  claude-sonnet-4-6
    ← structured JSON annotation
        │
        ▼
  append to tagged_output.csv
```

The model receives a metadata dict and returns a JSON object with seven annotation fields. No image bytes are sent — this pipeline operates purely on metadata, which is the reality of most large-scale survey workflows where images are archived separately from the tabular records.

---

## Prompt Design

The system prompt establishes the model's role as a NOAA SWFSC field analyst and specifies the exact annotation schema it must return. Keeping it stable across every record in a batch is intentional — the Anthropic API caches it after the first call, cutting input token costs by ~90% on batches of 50+ records.

```python
SYSTEM_PROMPT = """\
You are an expert marine wildlife ecologist and field survey analyst supporting \
NOAA's Southwest Fisheries Science Center (SWFSC) Ecosystem Science Division. \
...

## Critical rules
- Return ONLY a valid JSON object — no prose, no markdown fences, no commentary.
- quality_score must be an integer 1–5.
- confidence_level must be exactly "High", "Medium", or "Low".
- quality_assessment must be exactly "Excellent", "Good", "Marginal", or "Poor".\
"""
```

The user message formats a single record as a structured prompt and specifies the exact JSON shape to return:

```python
def build_user_prompt(record: dict) -> str:
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
}}"""
```

---

## Retry Logic

The API layer wraps every call in an exponential backoff loop that handles rate limits and server errors without crashing the pipeline. Rate limit responses include a `retry-after` header — reading it directly gives a more accurate wait time than a fixed backoff formula.

```python
for attempt in range(MAX_RETRIES):
    try:
        response = client.messages.create(...)
        return json.loads(response.content[0].text.strip())

    except anthropic.RateLimitError as exc:
        retry_after = float(
            exc.response.headers.get("retry-after", BASE_RETRY_DELAY * (2 ** attempt))
        )
        time.sleep(retry_after + random.uniform(0, 1))

    except anthropic.APIStatusError as exc:
        if exc.status_code >= 500:
            time.sleep(BASE_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1))
        else:
            raise  # 4xx errors are not retryable
```

Records that exhaust all retries are written to the output CSV with `ecological_classification = "ERROR"` and `quality_score = 1`, so the file stays complete and downstream steps can filter rather than crash.

---

## Output

Running the pipeline against 15 synthetic California Current survey records produces annotations like these:

**High-confidence cetacean ID — Monterey Bay humpback**

> *Megaptera novaeangliae observed via aerial drone at the surface in Monterey Bay (36.7954°N, 121.9612°W) on 2024-03-15. Timing and location are consistent with documented spring migratory corridor and known foraging grounds in the California Current System. No data quality flags.*

`quality_score: 5 | confidence: High | habitat: open water / pelagic`

---

**Ambiguous species ID — glider-based dolphin at depth**

The model correctly flagged a `Common Dolphin` record from a glider at 22 m as ambiguous between *Delphinus delphis* and *Delphinus capensis*:

> *Short-beaked Common Dolphin (Delphinus delphis) recorded at 22 m depth via underwater glider off Central CA Coast (36.4902°N, 122.9134°W) on 2024-03-18. D. capensis cannot be excluded without morphological imagery review. Glider platforms have limited capacity to visually confirm cetacean species identity at depth.*

`quality_score: 3 | confidence: Medium | habitat: open water / pelagic`

This is the right call — offshore pelagic position at that latitude favors *D. delphis*, but a glider can't confirm it, and the downstream model should know that.

---

**Unidentified towed camera record — quality floor**

A towed camera record at 55 m with no species ID came back with candidate taxa and a quality score of 2:

> *Species identity could not be resolved from available metadata, limiting utility for SDM and stock assessment pipelines. Recommend manual review by taxonomic expert before inclusion in downstream ML workflows.*

`quality_score: 2 | confidence: Low | habitat: rocky reef / benthic soft-sediment transition`

---

The full output CSV has 17 columns — the 8 original metadata fields plus 9 appended annotation columns:

| Column | Description |
|---|---|
| `ecological_classification` | Validated species in taxonomic nomenclature |
| `habitat_type` | Inferred habitat category |
| `behavior_notes` | Probable behavioral state or "insufficient metadata" |
| `quality_assessment` | Excellent / Good / Marginal / Poor |
| `quality_score` | Integer 1–5 for programmatic filtering |
| `survey_notes` | 1–2 sentence field database entry |
| `confidence_level` | High / Medium / Low |
| `processed_at` | UTC timestamp of annotation |
| `model_used` | Model version string |

---

## Scaling

This pipeline processes records sequentially, which is fine up to a few hundred records. Beyond that, the natural path is the [Anthropic Batches API](https://docs.anthropic.com/en/api/creating-message-batches) — submit the entire survey batch as a single job, poll for completion, write results in one pass. Batch processing cuts costs by 50% and eliminates rate-limit exposure entirely.

For real-time triage during an active deployment, `concurrent.futures.ThreadPoolExecutor` with concurrency tuned to the API rate-limit tier handles parallelization without adding a queue dependency.

---

## Running It

```bash
conda env create -f environment.yml && conda activate marine-tagger
export ANTHROPIC_API_KEY=sk-ant-...
python marine_tagger.py --input example_input.csv
```

The output CSV appends across runs, so successive batches accumulate into the same file. Add `--log-level DEBUG` to see per-record token usage and cache hit rates.

---

## What This Isn't

This pipeline annotates *metadata*, not images. It doesn't do computer vision — there's no base64 encoding of image bytes, no multimodal model call. The value is in taking the structured context that surrounds an image (where, when, what platform, what depth) and generating annotations that would otherwise require a trained analyst to produce manually.

The intended use case is pre-processing: run this against a survey batch before images enter a CV pipeline, flag low-quality records for manual review, and let the high-confidence annotations flow straight through to modeling. It reduces analyst burden without replacing expert judgment on the edge cases.

---

*Stack: Python 3.11, anthropic 0.105, pandas 3.0. Source at [https://github.com/imsibaja/marine-imagery-tagger](https://github.com/imsibaja/marine-imagery-tagger).*
