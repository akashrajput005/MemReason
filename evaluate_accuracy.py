import torch
import pickle
from torch.utils.data import DataLoader
from utils import get_stories, BabiDataset
from model import MemNN

def evaluate():
    with open('models/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    word2idx = metadata['word2idx']
    idx2word = metadata['idx2word']
    max_story_len = metadata['max_story_len']
    max_sent_len = metadata['max_sent_len']
    vocab_size = metadata['vocab_size']
    
    test_stories = get_stories('data/qa2_two-supporting-facts_test.txt')
    
    def pad_collate(batch):
        stories, queries, answers = zip(*batch)
        s_pad = torch.zeros((len(batch), max_story_len, max_sent_len), dtype=torch.long)
        for i, story in enumerate(stories):
            for j, sent in enumerate(story):
                if j < max_story_len:
                    s_pad[i, j, :len(sent)] = torch.tensor(sent)
        q_pad = torch.zeros((len(batch), max_sent_len), dtype=torch.long)
        for i, query in enumerate(queries):
            q_pad[i, :len(query)] = torch.tensor(query)
        a_tensor = torch.tensor(answers, dtype=torch.long)
        return s_pad, q_pad, a_tensor

    test_loader = DataLoader(BabiDataset(test_stories, word2idx, max_story_len, max_sent_len), 
                             batch_size=32, shuffle=False, collate_fn=pad_collate)

    model = MemNN(vocab_size, 128, max_sent_len, max_story_len=max_story_len, hop_count=3)
    model.load_state_dict(torch.load('models/memnn_model.pth', map_location=torch.device('cpu')))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for s, q, a in test_loader:
            outputs, _ = model(s, q)
            _, predicted = torch.max(outputs.data, 1)
            total += a.size(0)
            correct += (predicted == a).sum().item()
    
    print(f'Final Accuracy on Task 2 Test Set: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    evaluate()
