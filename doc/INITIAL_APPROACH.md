# Initial Project Approach

This document outlines the baseline approach we started with, following the core logic from the YouTube video.

## Baseline Technology Stack
- **Language**: Python
- **Framework**: Keras (TensorFlow backend)
- **Dataset**: bAbI (Task 1: Single Supporting Fact)

## Initial Architecture
The original approach focused on a basic Functional API model in Keras:
- **Dual Inputs**: One for the story and one for the question.
- **Simple Embeddings**: Converting tokens to vectors without weight tying.
- **Merging**: Using dot products to simulate a single hop of attention.
- **Output**: A softmax layer predicting the most likely answer from the vocabulary.

## Limitations of the Initial Approach
1. **Compatibility**: Standard TensorFlow/Keras struggled with the latest Python 3.14 environment on Windows.
2. **Reasoning Depth**: A single-hop approach is suitable for simple facts but fails at complex, multi-step reasoning (e.g., "Mary moved to X, then moved to Y").
3. **Explainability**: The original model was a "black box"—it gave an answer but didn't show which sentences it was looking at.
4. **Generalization**: Lacked modern regularization techniques like Layer Normalization and Dropout, making it prone to overfitting on small datasets.
