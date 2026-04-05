# MemReason: End-To-End Memory Network

An End-to-End Memory Network (MemNN) built in PyTorch for multi-hop reasoning and question answering on the bAbI dataset.

## Features
- **3-Hop Attention Mechanism**: Queries memory across multiple passes to perform multi-step reasoning.
- **Position & Temporal Encodings**: Correctly order facts in memory and incorporate chronological data context.
- **Weight Tying (Type A)**: Parameter sharing across hops for more stable gradients and faster convergence.
- **Reasoning logic visualization**: Built-in visualizer to print out the model's exact attention weights and logic trace on a per-sentence basis.

## Running the Chatbot
```bash
python chatbot.py 
```

## Running the Training Pipeline
To retrain the model on the bAbI dataset (Task 2):
```bash
python train.py
```

## Relevant Reading
See `doc/resume_tech_explanations.md` for an in-depth breakdown of the concepts used in this codebase.
