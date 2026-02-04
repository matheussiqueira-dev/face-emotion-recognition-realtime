import uvicorn
import argparse
from app.backend.api import app

def main():
    parser = argparse.ArgumentParser(description="Run the Face Emotion Recognition Web API")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    
    args = parser.parse_args()
    
    print(f"\n🚀 EmotionAI is starting at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server.\n")
    
    uvicorn.run("app.backend.api:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
