import argparse
from dataclasses import replace
from pathlib import Path

from src.config import Config
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding-powered data cleansing")
    parser.add_argument("--threshold", type=float, help="Similarity threshold override")
    parser.add_argument("--top-k", type=int, help="Top-K neighbors override")
    parser.add_argument("--collection", type=str, help="Qdrant collection name override")
    return parser.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    updates = {}
    if args.threshold is not None:
        updates["sim_threshold"] = args.threshold
    if args.top_k is not None:
        updates["top_k"] = args.top_k
    if args.collection is not None:
        updates["collection_name"] = args.collection
    if not updates:
        return config
    return replace(config, **updates)


def print_config_summary(config: Config) -> None:
    api_key_status = "set" if config.openai_api_key else "missing"
    print("Config summary:")
    print(f"  EMBED_MODEL: {config.embed_model}")
    print(f"  OPENAI_API_KEY: {api_key_status}")
    print(f"  OLLAMA_ENDPOINT: {config.ollama_endpoint}")
    print(f"  QDRANT_URL: {config.qdrant_url}")
    print(f"  SIM_THRESHOLD: {config.sim_threshold}")
    print(f"  TOP_K: {config.top_k}")
    print(f"  COLLECTION_NAME: {config.collection_name}")


def main() -> None:
    args = parse_args()
    config = apply_overrides(Config.from_env(), args)
    print_config_summary(config)
    run_pipeline(
        config=config,
        data_path=Path("data/companies_raw.csv"),
        report_path=Path("report.html"),
        gold_path=Path("data/companies_gold.csv"),
        log=print,
    )


if __name__ == "__main__":
    main()
