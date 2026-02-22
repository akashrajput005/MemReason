# Setup & Run Guide

Follow these steps to set up the environment and run the chatbot.

## 1. Environment Setup
Ensure you have Python 3.10+ installed. Install the required dependencies:
```powershell
pip install torch numpy
```

## 2. Running the Interactive Chatbot
To start chatting with MindTrace and see the reasoning analysis:
```powershell
python chatbot.py
```
### How to use:
- **Story**: Enter sentences separated by dots (e.g., `Mary is in the kitchen. John went to the garden.`)
- **Question**: Ask a question about the story (e.g., `Where is Mary?`)
- **Reasoning Analysis**: Observe the attention bars (`====`) to see which facts the model prioritized.

## 3. Running the Demo Script
For a quick automated demonstration of the reasoning logic:
```powershell
python demo_reasoning.py
```

## 4. Training (Advanced)
If you wish to retrain the model on different tasks:
1. Update any hyperparameters in `train.py`.
2. Run:
```powershell
python train.py
```
