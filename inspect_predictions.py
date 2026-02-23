import torch
import pickle
from model import MemNN
from utils import get_stories, tokenize

def inspect():
    with open('models/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    word2idx = metadata['word2idx']
    idx2word = metadata['idx2word']
    max_story_len = metadata['max_story_len']
    max_sent_len = metadata['max_sent_len']
    vocab_size = metadata['vocab_size']
    
    test_stories = get_stories('data/qa2_two-supporting-facts_test.txt')
    
    model = MemNN(vocab_size, 128, max_sent_len, max_story_len=max_story_len, hop_count=3)
    model.load_state_dict(torch.load('models/memnn_model.pth', map_location=torch.device('cpu')))
    model.eval()

    print("\n--- Inspecting First 5 Test Cases ---\n")
    for i in range(5):
        story, query, answer = test_stories[i]
        
        # Vectorize
        s_vec = torch.zeros((1, max_story_len, max_sent_len), dtype=torch.long)
        for j, sent in enumerate(story):
            if j < max_story_len:
                for k, word in enumerate(sent):
                    if k < max_sent_len:
                        s_vec[0, j, k] = word2idx.get(word, 0)
        
        q_vec = torch.zeros((1, max_sent_len), dtype=torch.long)
        for j, word in enumerate(query):
            if j < max_sent_len:
                q_vec[0, j] = word2idx.get(word, 0)
        
        with torch.no_grad():
            outputs, all_probs = model(s_vec, q_vec)
            _, predicted = torch.max(outputs, 1)
            pred_word = idx2word.get(predicted.item(), "Unknown")
        
        print(f"Case {i+1}:")
        print(f"Story: {' '.join([' '.join(s) for s in story[:3]])} ...")
        print(f"Question: {' '.join(query)}")
        print(f"True Answer: {answer}")
        print(f"Predicted: {pred_word}")
        
        # Show attention for hops
        for hop_idx, probs in enumerate(all_probs):
            top_idx = torch.argmax(probs[0]).item()
            if top_idx < len(story):
                top_sent = " ".join(story[top_idx])
                print(f"  Hop {hop_idx+1} Top Attn: {probs[0, top_idx].item():.2f} -> {top_sent}")
        print("-" * 20)

if __name__ == "__main__":
    inspect()
