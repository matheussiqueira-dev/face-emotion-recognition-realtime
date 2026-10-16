import argparse

import uvicorn

from app.core.config import AppConfig


def main():
    parser = argparse.ArgumentParser(description="Run the Face Emotion Recognition Web API")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")

    args = parser.parse_args()

    print(f"\nEmotionAI is starting at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server.\n")

    uvicorn.run(
        "app.backend.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        ws_max_size=AppConfig().max_frame_bytes + 4096,
        ws_max_queue=2,
    )


if __name__ == "__main__":
    main()
