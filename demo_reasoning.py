import torch
import pickle
import os
from utils import tokenize, get_stories
from model import MemNN

def demo():
    with open('models/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    word2idx = metadata['word2idx']
    idx2word = metadata['idx2word']
    max_story_len = metadata['max_story_len']
    max_sent_len = metadata['max_sent_len']
    vocab_size = metadata['vocab_size']
    
    model = MemNN(vocab_size, 128, max_sent_len, hop_count=3)
    model.load_state_dict(torch.load('models/memnn_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    # Use a real case from Task 2 data
    story_input = "Mary moved to the bathroom. Sandra journeyed to the bedroom. Mary moved to the hallway. Mary travelled to the office. Sandra move to the garden."
    question_input = "Where is Mary?"
    
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
        outputs, all_probs = model(s_vec, q_vec)
        _, predicted = torch.max(outputs, 1)
        answer = idx2word.get(predicted.item(), "Unknown")
        
    print(f"Story: {story_input}")
    print(f"Question: {question_input}")
    print(f"\n--- Reasoning Analysis ---")
    last_hop_attention = all_probs[-1][0].cpu().numpy()
    
    for i, sent in enumerate(story_sentences):
        attn = last_hop_attention[i]
        bar = "=" * int(attn * 20)
        text = " ".join(sent)
        print(f"{attn:.2f} {bar:20} | {text}")
        
    print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    demo()
