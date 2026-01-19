# Data Dedupe

Embeddings + similarity search for fast, explainable data deduplication.

## What it does
- Embeds company names (OpenAI or Ollama).
- Finds near-duplicates with Qdrant.
- Clusters matches and picks canonicals.
- Generates `report.html`.

## Run with Docker (recommended)
1. Copy `.env.example` to `.env`.
2. Set one embedding provider:
   - OpenAI: set `OPENAI_API_KEY` and optionally `EMBED_MODEL`.
   - Ollama: set `OLLAMA_ENDPOINT` and `EMBED_MODEL`, leave `OPENAI_API_KEY` empty.
3. Start the stack:
   ```bash
   docker compose up --build
   ```
4. Open the app at `http://localhost:8000`.

## LinkedIn quick demo
1. Open the app and upload `data/companies_raw.csv`.
2. Leave defaults or try `SIM_THRESHOLD=0.86`, `TOP_K=5`.
3. Click **Generate report** and open `report.html`.
4. Share a screenshot of the clusters + report summary.

## Input format
CSV must include:
- `id`
- `company_name`

## Configuration
Environment variables (all optional unless noted):
- `OPENAI_API_KEY` (required for OpenAI)
- `OLLAMA_ENDPOINT` (required for Ollama)
- `EMBED_MODEL` (defaults in `.env.example`)
- `QDRANT_URL` (default: `http://qdrant:6333`)
- `SIM_THRESHOLD`
- `TOP_K`
- `COLLECTION_NAME`

## Notes
- If you change embedding models, use a new `COLLECTION_NAME` or delete the old Qdrant collection.
- `report.html` is generated in the repo root.

## License
MIT
