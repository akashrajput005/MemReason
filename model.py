import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def position_encoding(sentence_size, embedding_dim):
    """
    Position Encoding as described in the End-to-End Memory Networks paper.
    """
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size
    le = embedding_dim
    for it in range(1, le + 1):
        for jt in range(1, ls + 1):
            res = (1 - jt / ls) - (it / le) * (1 - 2 * jt / ls)
            encoding[it - 1, jt - 1] = res
    return torch.from_numpy(encoding).T  # (sentence_size, embedding_dim)

class MemNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_sent_len, max_story_len=100, hop_count=3, dropout=0.5):
        super(MemNN, self).__init__()
        self.hop_count = hop_count
        self.embedding_dim = embedding_dim
        
        self.pe = position_encoding(max_sent_len, embedding_dim)
        
        # Weight Tying Type A: A_k+1 = C_k, B = A_0, W = C_K
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim, padding_idx=0) for _ in range(hop_count + 1)])
        
        # Temporal Encodings
        self.T = nn.ModuleList([nn.Embedding(max_story_len, embedding_dim) for _ in range(hop_count + 1)])
        
        self.dropout = nn.Dropout(dropout)
        # Removed LayerNorm to prevent complexity-based overfitting
        
        # Initialization
        for emb in self.embeddings:
            nn.init.normal_(emb.weight.data, std=0.1)
            emb.weight.data[0].fill_(0)
        for emb in self.T:
            nn.init.normal_(emb.weight.data, std=0.1)

    def forward(self, stories, queries):
        device = stories.device
        batch_size, num_sentences, _ = stories.size()
        pe = self.pe.to(device)
        
        # Reverse Temporal Indices
        t_indices = torch.arange(num_sentences, device=device).flip(0).unsqueeze(0).repeat(batch_size, 1)
        
        # Query B = A_0
        u = (self.embeddings[0](queries) * pe[:queries.size(1), :]).sum(dim=1)
        u = self.dropout(u)
        
        s_mask = stories.sum(dim=2) != 0
        
        all_probs = []
        for k in range(self.hop_count):
            m = (self.embeddings[k](stories) * pe).sum(dim=2) + self.T[k](t_indices)
            m = self.dropout(m)
            
            c = (self.embeddings[k+1](stories) * pe).sum(dim=2) + self.T[k+1](t_indices)
            c = self.dropout(c)
            
            attn_scores = torch.bmm(u.unsqueeze(1), m.transpose(1, 2)).squeeze(1)
            # Scale scores slightly to keep softmax from being too sharp during high dropout
            attn_scores = attn_scores / np.sqrt(self.embedding_dim)
            
            masked_scores = attn_scores.masked_fill(~s_mask, -1e9)
            probs = F.softmax(masked_scores, dim=-1).unsqueeze(1)
            all_probs.append(probs.squeeze(1))
            
            o = torch.bmm(probs, c).squeeze(1)
            
            # Pure update
            u = u + o
            
        # Logits W = A_last
        logits = F.linear(u, self.embeddings[-1].weight)
        return logits, all_probs
