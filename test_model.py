import torch
import pickle
from utils import tokenize
from model import MemNN

def test_model():
    with open('models/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    word2idx = metadata['word2idx']
    idx2word = metadata['idx2word']
    max_story_len = metadata['max_story_len']
    max_sent_len = metadata['max_sent_len']
    vocab_size = metadata['vocab_size']
    
    model = MemNN(vocab_size, 64, max_sent_len, hop_count=3)
    model.load_state_dict(torch.load('models/memnn_model.pth', map_location=torch.device('cpu')))
    model.eval()

    test_cases = [
        {
            "story": "Mary moved to the bathroom. John went to the hallway.",
            "question": "Where is Mary?"
        },
        {
            "story": "Daniel went back to the hallway. Sandra moved to the garden.",
            "question": "Where is Daniel?"
        }
    ]

    for case in test_cases:
        story_input = case["story"]
        question_input = case["question"]
        
        story_sentences = [tokenize(s.strip() + ".") for s in story_input.split('.') if s.strip()]
        question_tokens = tokenize(question_input)
        
        s_vec = torch.zeros((1, max_story_len, max_sent_len), dtype=torch.long)
        for i, sent in enumerate(story_sentences):
            if i < max_story_len:
                for j, word in enumerate(sent):
                    if j < max_sent_len:
                        s_vec[0, i, j] = word2idx.get(word, 0)
        
        q_vec = torch.zeros((1, max_sent_len), dtype=torch.long)
        for i, word in enumerate(question_tokens):
            if i < max_sent_len:
                q_vec[0, i] = word2idx.get(word, 0)
        
        with torch.no_grad():
            outputs = model(s_vec, q_vec)
            _, predicted = torch.max(outputs, 1)
            answer = idx2word.get(predicted.item(), "Unknown")
            
        print(f"Story: {story_input}")
        print(f"Question: {question_input}")
        print(f"Predicted Answer: {answer}\n")

if __name__ == "__main__":
    test_model()
