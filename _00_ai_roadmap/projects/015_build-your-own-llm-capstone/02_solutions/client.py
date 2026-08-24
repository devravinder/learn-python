"""A tiny client for serve_api.py - Stage 4's round-trip proof.

Usage:
    python client.py "Continue this story:"
"""
import sys

import requests

if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "Continue this story:"
    response = requests.post("http://localhost:8000/chat", json={"message": message})
    print("Model response:", response.json()["response"])
