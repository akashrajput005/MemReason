# Enhanced Version: Next Level Reasoning

This document details the advanced updates and architectural improvements that took the chatbot beyond the initial baseline.

## 1. Migration to PyTorch
We pivoted the entire codebase to **PyTorch**. This allowed for:
- Seamless compatibility with **Python 3.14**.
- More granular control over the memory network's internal states.
- Faster development of custom layers for reasoning.

## 2. Multi-hop Reasoning Logic
Instead of a single "look" at the data, the enhanced model implements **3 Hops** of reasoning:
- **Hop 1**: Identify the primary subject and location.
- **Hop 2**: Find secondary associations (e.g., who was with them).
- **Hop 3**: Final verification of the logical state (e.g., current location after movements).

## 3. Position Encoding
We implemented **Position Encoding (PE)** as described in the FAIR Research papers. This allows the model to "know" which events happened recently and which happened long ago, which is critical for reasoning about a sequence of movements.

## 4. Logic Explainability (Attention Visualization)
The "next level" feature is the **Reasoning Analysis**. Every time the chatbot answers, it prints a list of sentences from the story with a visual bar:
- **Weight 0.80+**: High focus, primary fact used for the answer.
- **Weight 0.20-0.50**: Secondary context or supporting evidence.
- **Weight < 0.10**: Ignored information.

## 5. Deployment for 'Hard Mode'
The model was scaled to handle **Task 2 (Two Supporting Facts)**. This task is significantly more difficult than the baseline, as it requires the model to link two different facts to reach a conclusion (e.g., "John is in the kitchen. The milk is with John. Where is the milk?").

## 6. Model Polishing
We added **Layer Normalization** and **Dropout** (0.2), which stabilized the training and allowed for a larger embedding size (128) without overfitting the small stories.
