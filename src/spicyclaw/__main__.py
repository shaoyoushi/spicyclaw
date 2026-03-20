"""Entry point: parse args and start uvicorn."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from spicyclaw.config import Settings


def _setup_logging(log_level: str, logs_dir: Path) -> None:
    """Configure console + file logging."""
    logs_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(logs_dir / "spicyclaw.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="SpicyClaw AI Agent")
    parser.add_argument("--host", default=None, help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="Bind port")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--api-base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--lang", default=None, help="Language: en or zh")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()

    # Build settings overrides from CLI args
    overrides: dict = {}
    if args.host:
        overrides["host"] = args.host
    if args.port:
        overrides["port"] = args.port
    if args.model:
        overrides["model"] = args.model
    if args.api_base_url:
        overrides["api_base_url"] = args.api_base_url
    if args.api_key:
        overrides["api_key"] = args.api_key
    if args.lang:
        overrides["lang"] = args.lang

    settings = Settings(**overrides)

    _setup_logging(args.log_level, settings.logs_dir)

    from spicyclaw.common.i18n import set_lang
    set_lang(settings.lang)

    from spicyclaw.gateway.server import create_app

    app = create_app(settings)

    uvicorn.run(app, host=settings.host, port=settings.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
