from __future__ import annotations

import os
import uuid
from dataclasses import replace
from pathlib import Path

from flask import Flask, Response, request

from src.config import Config
from src.pipeline import run_pipeline


UPLOAD_DIR = Path("/tmp/embeddings_uploads")
REPORT_DIR = Path("/tmp/embeddings_reports")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def _render_layout(content: str, title: str = "Data Dedupe") -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;600&display=swap');

    :root {{
      --navy-900: #0d1026;
      --navy-800: #141a3a;
      --navy-700: #1b2451;
      --ink: #e8edf7;
      --muted: rgba(232, 237, 247, 0.7);
      --purple: #7b5cff;
      --magenta: #d62468;
      --red: #e11d48;
      --orange: #f59e42;
      --card: rgba(16, 20, 44, 0.82);
      --border: rgba(123, 92, 255, 0.25);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(123, 92, 255, 0.18), transparent 55%),
        radial-gradient(circle at 20% 80%, rgba(225, 29, 72, 0.16), transparent 50%),
        linear-gradient(135deg, var(--navy-900) 0%, var(--navy-800) 45%, var(--navy-700) 100%);
      min-height: 100vh;
    }}

    .hero {{
      padding: 64px 16px 32px;
      text-align: center;
      position: relative;
      overflow: hidden;
    }}

    .orb {{
      position: absolute;
      border-radius: 999px;
      filter: blur(0px);
      opacity: 0.5;
      animation: float 10s ease-in-out infinite;
    }}

    .orb.one {{
      width: 220px;
      height: 220px;
      background: rgba(123, 92, 255, 0.2);
      top: -60px;
      left: -40px;
    }}

    .hero h1 {{
      font-size: clamp(2rem, 4vw, 3.2rem);
      margin-bottom: 12px;
      letter-spacing: -0.02em;
    }}

    .hero p {{
      margin: 0 auto;
      max-width: 680px;
      color: var(--muted);
      font-size: 1.05rem;
    }}

    .card {{
      max-width: 820px;
      margin: -24px auto 48px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 20px 50px rgba(6, 9, 25, 0.6);
      backdrop-filter: blur(6px);
      animation: rise 0.8s ease-out;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-top: 20px;
    }}

    label {{
      font-weight: 600;
      font-size: 0.9rem;
    }}

    input[type=\"file\"],
    input[type=\"number\"],
    input[type=\"text\"] {{
      width: 100%;
      padding: 10px 12px;
      margin-top: 6px;
      border-radius: 10px;
      border: 1px solid rgba(123, 92, 255, 0.35);
      background: rgba(255, 255, 255, 0.92);
      font-family: "IBM Plex Sans", sans-serif;
    }}

    .btn {{
      margin-top: 20px;
      padding: 12px 20px;
      border-radius: 999px;
      border: none;
      background: linear-gradient(120deg, var(--purple), var(--magenta), var(--orange));
      color: white;
      font-weight: 600;
      font-size: 1rem;
      cursor: pointer;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      width: 100%;
    }}

    .btn:hover {{
      transform: translateY(-1px);
      box-shadow: 0 12px 28px rgba(123, 92, 255, 0.35);
    }}

    .hint {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.9rem;
    }}

    .error {{
      margin-top: 16px;
      color: #ffd3dc;
      background: rgba(225, 29, 72, 0.18);
      border: 1px solid rgba(225, 29, 72, 0.4);
      padding: 12px;
      border-radius: 12px;
      white-space: pre-wrap;
    }}

    .logs {{
      margin-top: 18px;
      padding: 12px;
      background: rgba(12, 18, 40, 0.85);
      color: #e2e8f0;
      border-radius: 12px;
      font-family: "IBM Plex Sans", sans-serif;
      font-size: 0.85rem;
      white-space: pre-wrap;
    }}

    .footer {{
      margin: 0 auto 40px;
      text-align: center;
      color: var(--muted);
      font-size: 0.95rem;
    }}

    .footer img {{
      display: block;
      max-width: min(360px, 80vw);
      width: 100%;
      height: auto;
      margin: 0 auto;
    }}

    @keyframes rise {{
      from {{ transform: translateY(12px); opacity: 0; }}
      to {{ transform: translateY(0); opacity: 1; }}
    }}

    @keyframes float {{
      0%, 100% {{ transform: translateY(0px); }}
      50% {{ transform: translateY(10px); }}
    }}
  </style>
</head>
<body>
  <section class=\"hero\">
    <div class=\"orb one\"></div>
    <h1>Data Dedupe</h1>
    <p>Embeddings + similarity search for fast, explainable data deduplication.</p>
  </section>
  <div class=\"card\">
    {content}
  </div>
  <footer class=\"footer\">
    <a href=\"https://www.strategyagents.com\" target=\"_blank\" rel=\"noopener\">
      <img src=\"https://s3.us-east-1.amazonaws.com/strategyagents.com/img/arrow/SA_logo+color_reverse.png\" alt=\"Strategy Agents\" />
    </a>
  </footer>
</body>
</html>
"""


def _render_form(error: str | None = None, logs: str | None = None) -> str:
    error_block = f"<div class=\"error\">{error}</div>" if error else ""
    logs_block = f"<div class=\"logs\">{logs}</div>" if logs else ""
    return _render_layout(
        f"""
        <form method=\"post\" action=\"/run\" enctype=\"multipart/form-data\">
          <label for=\"file\">Variations CSV</label>
          <input id=\"file\" name=\"file\" type=\"file\" accept=\".csv\" required />
          <div class=\"hint\">Expected columns: <strong>id</strong>, <strong>company_name</strong>.</div>

          <div class=\"grid\">
            <div>
              <label for=\"threshold\">Similarity threshold</label>
              <input id=\"threshold\" name=\"threshold\" type=\"number\" step=\"0.01\" min=\"0\" max=\"1\" placeholder=\"0.83\" />
            </div>
            <div>
              <label for=\"top_k\">Top-K neighbors</label>
              <input id=\"top_k\" name=\"top_k\" type=\"number\" min=\"1\" max=\"50\" placeholder=\"5\" />
            </div>
            <div>
              <label for=\"collection\">Collection name</label>
              <input id=\"collection\" name=\"collection\" type=\"text\" placeholder=\"companies\" />
            </div>
          </div>

          <div class=\"grid\">
            <div>
              <label for=\"use_master\">
                <input id=\"use_master\" name=\"use_master\" type=\"checkbox\" />
                Match to master list
              </label>
              <div class=\"hint\">When enabled, clusters are named using the closest master record.</div>
            </div>
            <div>
              <label for=\"master_file\">Master CSV (optional)</label>
              <input id=\"master_file\" name=\"master_file\" type=\"file\" accept=\".csv\" />
              <div class=\"hint\">Required if \"Match to master list\" is checked.</div>
            </div>
          </div>

          <button class=\"btn\" type=\"submit\">Generate report</button>
        </form>
        {error_block}
        {logs_block}
        """,
        title="Embedding Dedupe Studio",
    )


@app.get("/")
def index() -> Response:
    return Response(_render_form(), mimetype="text/html")


@app.post("/run")
def run_upload() -> Response:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return Response(_render_form(error="Please select a CSV file."), mimetype="text/html")

    use_master = request.form.get("use_master") == "on"
    master_upload = request.files.get("master_file")
    if use_master and (master_upload is None or master_upload.filename == ""):
        return Response(
            _render_form(error="Master CSV is required when matching to a master list."),
            mimetype="text/html",
        )

    upload_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{upload_id}.csv"
    uploaded.save(upload_path)

    master_path = None
    if use_master and master_upload is not None:
        master_path = UPLOAD_DIR / f"{upload_id}_master.csv"
        master_upload.save(master_path)

    base_config = Config.from_env()
    threshold = _parse_float(request.form.get("threshold"))
    top_k = _parse_int(request.form.get("top_k"))
    collection = request.form.get("collection")
    collection_name = collection.strip() if collection else ""
    if not collection_name:
        collection_name = f"{base_config.collection_name}_{upload_id[:8]}"

    config = replace(
        base_config,
        sim_threshold=threshold if threshold is not None else base_config.sim_threshold,
        top_k=top_k if top_k is not None else base_config.top_k,
        collection_name=collection_name,
    )

    report_path = REPORT_DIR / f"report_{upload_id}.html"
    logs: list[str] = []

    try:
        run_pipeline(
            config=config,
            data_path=upload_path,
            report_path=report_path,
            gold_path=None,
            master_path=master_path,
            log=logs.append,
        )
    except Exception as exc:  # noqa: BLE001
        logs_text = "\n".join(logs)
        return Response(
            _render_form(error=f"{exc}", logs=logs_text),
            mimetype="text/html",
        )

    report_html = report_path.read_text(encoding="utf-8")
    return Response(report_html, mimetype="text/html")


def _parse_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
