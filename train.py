import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import os
from utils import get_stories, BabiDataset, tokenize
from model import MemNN

# Hyperparameters
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
HOPS = 3

def train():
    # Load data - Switching to Task 2 (Two Supporting Facts)
    train_stories = get_stories('data/qa2_two-supporting-facts_train.txt')
    test_stories = get_stories('data/qa2_two-supporting-facts_test.txt')

    # Build vocabulary
    vocab = set()
    for story, query, answer in train_stories + test_stories:
        vocab.update([word for sent in story for word in sent] + query + [answer])
    
    vocab = sorted(vocab)
    vocab_size = len(vocab) + 1  # +1 for padding
    word2idx = {word: i + 1 for i, word in enumerate(vocab)}
    idx2word = {i + 1: word for i, word in enumerate(vocab)}

    # Max lengths for padding
    max_story_len = max([len(story) for story, _, _ in train_stories + test_stories])
    max_sent_len = max([len(sent) for story, _, _ in train_stories + test_stories for sent in story] + 
                       [len(query) for _, query, _ in train_stories + test_stories])
    
    print(f"Vocab size: {vocab_size}")
    print(f"Max story length (sentences): {max_story_len}")
    print(f"Max sentence length: {max_sent_len}")

    # Helper function to pad sequences
    def pad_collate(batch):
        stories, queries, answers = zip(*batch)
        
        # S_pad: (batch, max_story_len, max_sent_len)
        s_pad = torch.zeros((len(batch), max_story_len, max_sent_len), dtype=torch.long)
        for i, story in enumerate(stories):
            for j, sent in enumerate(story):
                if j < max_story_len:
                    s_pad[i, j, :len(sent)] = torch.tensor(sent)
        
        # Q_pad: (batch, max_sent_len)
        q_pad = torch.zeros((len(batch), max_sent_len), dtype=torch.long)
        for i, query in enumerate(queries):
            q_pad[i, :len(query)] = torch.tensor(query)
            
        a_tensor = torch.tensor(answers, dtype=torch.long)
        
        return s_pad, q_pad, a_tensor

    train_loader = DataLoader(BabiDataset(train_stories, word2idx, max_story_len, max_sent_len), 
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(BabiDataset(test_stories, word2idx, max_story_len, max_sent_len), 
                             batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MemNN(vocab_size, EMBEDDING_DIM, max_sent_len, HOPS).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for s, q, a in train_loader:
            s, q, a = s.to(device), q.to(device), a.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(s, q)
            loss = criterion(outputs, a)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 40.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            # Evaluation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for s, q, a in test_loader:
                    s, q, a = s.to(device), q.to(device), a.to(device)
                    outputs, _ = model(s, q)
                    _, predicted = torch.max(outputs.data, 1)
                    total += a.size(0)
                    correct += (predicted == a).sum().item()
            
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

    # Save model and metadata
    if not os.path.exists('models'):
        os.makedirs('models')
    
    torch.save(model.state_dict(), 'models/memnn_model.pth')
    with open('models/metadata.pkl', 'wb') as f:
        pickle.dump({'word2idx': word2idx, 'idx2word': idx2word, 'max_story_len': max_story_len, 'max_sent_len': max_sent_len, 'vocab_size': vocab_size}, f)
    
    print("Model and metadata saved to models/.")

if __name__ == "__main__":
    train()
