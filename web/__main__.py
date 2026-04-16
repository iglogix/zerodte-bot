"""CLI entry point: python -m web"""
from __future__ import annotations

import argparse
import uvicorn
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="0DTE Bot Web Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    uvicorn.run(
        "web.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
