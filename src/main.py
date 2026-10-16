"""CLI runner and Uvicorn entrypoint."""

import argparse
import os
import uvicorn
from dotenv import load_dotenv

from src.core import ChurnModelPipeline

load_dotenv()


def main() -> None:
    """Main CLI parser for running server or training models."""
    parser = argparse.ArgumentParser(description="Simple Customer Churn Predictor CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the machine learning model on synthetic data")
    train_parser.add_argument("--samples", type=int, default=2000, help="Number of synthetic training samples")
    train_parser.add_argument("--output", type=str, default="artifacts/churn_model.joblib", help="Path to save trained model")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI Uvicorn server")
    serve_parser.add_argument("--host", type=str, default=os.getenv("API_HOST", "0.0.0.0"))
    serve_parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "8000")))

    args = parser.parse_args()

    if args.command == "train":
        print(f"Generating {args.samples} synthetic training samples...")
        pipeline = ChurnModelPipeline(model_path=args.output)
        df, y = pipeline.generate_synthetic_data(n_samples=args.samples)
        metrics = pipeline.train(df, y)
        print("Training completed successfully!")
        print(f"Metrics: {metrics}")
        print(f"Model saved to {args.output}")
    elif args.command == "serve":
        host = args.host
        port = args.port
        print(f"Starting FastAPI server on {host}:{port}...")
        uvicorn.run("src.api:app", host=host, port=port, reload=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
