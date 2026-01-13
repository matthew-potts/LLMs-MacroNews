# LLMs-MacroNews

Lightweight pipeline to fetch macro news, create batch LLM requests, submit and monitor batch jobs, collect outputs, and convert them into analysis-ready CSVs.

This software was created to support the analysis in my recent research article on [LLM assessments of macroeconomic news and their relation to the Global Financial Cycle](https://www.crossbordercode.com/research/LLMsAndGFCy/LLMsAndGFCy.html).

---

Environment variables:
- `OPEN_AI_SECRET_KEY` (or whichever key your LLM client expects)
- Any other provider credentials if you use Refinitiv or other data sources

---

## Quick Start
1. Create and activate a virtualenv (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2. Set API keys (zsh/macOS):

```bash
export OPEN_AI_SECRET_KEY="sk-..."
```

3. Open the interactive notebook `analysis/BatchCreation.ipynb` and run cells to build requests, upload batch files, create batch jobs, and collect outputs.

4. Or run the workflow module with the provided CLI (if available):

```bash
python -m src.workflow.program --help
```

---

## Configuration
- `src/config/config.json` (if present) stores runtime flags and defaults used by the workflow.
- Logging configuration is in `src/logging/logger.py`.

---

## Core Components
- `src/lib/functions.py` — data helpers (fetching, timestamp handling, preparing regression data).
- `src/lib/batch/batch.py` — batching helpers (`create_jsonl`, `create_batches`, `write_results_from_batches`, `read_batch_outputs_to_df`).
- `src/lib/llm_client.py` and provider-specific clients (OpenAI/Anthropic/Nebius) provide a consistent LLM interface.
- `src/workflow/program.py` — top-level workflow orchestration and CLI entrypoint.

---

## Batch Flow (high-level)
1. Build a JSONL request file (one JSON object per line) using `create_jsonl`.
2. Split the JSONL into smaller batch files and upload them.
3. Create batch jobs referencing the uploaded files; poll until complete.
4. Download batch outputs and combine into a single JSONL using `write_results_from_batches`.
5. Parse the combined JSONL with `read_batch_outputs_to_df` to extract `requestID`, `storyID`, `content`, and write analysis CSVs.
---