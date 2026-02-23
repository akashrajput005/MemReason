import torch
import pickle
import os
from utils import tokenize
from model import MemNN

def load_model():
    if not os.path.exists('models/memnn_model.pth') or not os.path.exists('models/metadata.pkl'):
        print("Model or metadata not found. Please run train.py first.")
        return None, None
    
    with open('models/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    word2idx = metadata['word2idx']
    idx2word = metadata['idx2word']
    max_story_len = metadata['max_story_len']
    max_sent_len = metadata['max_sent_len']
    vocab_size = metadata['vocab_size']
    
    model = MemNN(vocab_size, 128, max_sent_len, max_story_len=max_story_len, hop_count=3)
    model.load_state_dict(torch.load('models/memnn_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return model, metadata

def chatbot():
    model, metadata = load_model()
    if not model:
        return
    
    word2idx = metadata['word2idx']
    idx2word = metadata['idx2word']
    max_story_len = metadata['max_story_len']
    max_sent_len = metadata['max_sent_len']
    
    print("\n--- bAbI Chatbot ---")
    print("Type your story followed by a question.")
    print("Example story: Mary moved to the bathroom. John went to the hallway.")
    print("Example question: Where is Mary?")
    print("Type 'exit' to quit.\n")
    
    while True:
        story_input = input("Story (sentences separated by dots): ")
        if story_input.lower() == 'exit':
            break
        
        question_input = input("Question: ")
        if question_input.lower() == 'exit':
            break
            
        # Preprocess
        story_sentences = [tokenize(s.strip() + ".") for s in story_input.split('.') if s.strip()]
        question_tokens = tokenize(question_input)
        
        # Vectorize
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
        
        # Predict
        with torch.no_grad():
            outputs, all_probs = model(s_vec, q_vec)
            _, predicted = torch.max(outputs, 1)
            answer = idx2word.get(predicted.item(), "Unknown")
            
        print(f"\n--- Reasoning Analysis ---")
        # all_probs is a list of [batch, num_sentences] for each hop
        # We'll show the last hop attention as it's usually the most focused
        last_hop_attention = all_probs[-1][0].cpu().numpy()
        
        for i, sent in enumerate(story_sentences):
            attn = last_hop_attention[i]
            bar = "=" * int(attn * 20)
            text = " ".join(sent)
            print(f"{attn:.2f} {bar:20} | {text}")
            
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    chatbot()
