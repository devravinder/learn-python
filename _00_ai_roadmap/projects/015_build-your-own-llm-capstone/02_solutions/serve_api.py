"""Stage 4: serve the instruction-tuned GPT behind a minimal chat API.
Lesson 078 (Module 15) covers FastAPI serving properly - this is a
working preview, kept intentionally small.

Usage:
    python serve_api.py --checkpoint gpt_instruction_tuned.pt --data ../../013_train-your-own-gpt/02_solutions/data/fallback_corpus.txt
    # then, in another terminal:
    python client.py "Continue this story:"
"""
import argparse
import sys
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_013 = Path(__file__).parents[2] / "013_train-your-own-gpt" / "02_solutions"
sys.path.insert(0, str(PROJECT_013))
from bpe_tokenizer import BPETokenizer  # noqa: E402
from model import GPT  # noqa: E402

app = FastAPI(title="Your Own LLM — Chat API")

_state = {}


class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 60
    temperature: float = 0.8


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    model, tokenizer, config = _state["model"], _state["tokenizer"], _state["config"]
    prompt = f"<|user|>\n{request.message}\n<|assistant|>\n"
    idx = torch.tensor([tokenizer.encode(prompt)])

    with torch.no_grad():
        for _ in range(request.max_new_tokens):
            logits = model(idx[:, -config["block_size"]:])[:, -1, :] / request.temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

    full_text = tokenizer.decode(idx[0].tolist())
    response_text = full_text.split("<|assistant|>\n", 1)[-1]
    return ChatResponse(response=response_text)


def load_model(checkpoint_path, data_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    tokenizer = BPETokenizer()
    tokenizer.load(Path(data_path).with_suffix(".tokenizer.json"))
    model = GPT(
        vocab_size=len(tokenizer.vocab), d_model=config["d_model"], n_heads=config["n_heads"],
        n_layers=config["n_layers"], d_ff=config["d_ff"], max_len=config["block_size"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, config


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="gpt_instruction_tuned.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model, tokenizer, config = load_model(args.checkpoint, args.data)
    _state.update({"model": model, "tokenizer": tokenizer, "config": config})

    uvicorn.run(app, host="0.0.0.0", port=args.port)
