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
    def __init__(self, vocab_size, embedding_dim, max_sent_len, hop_count=3, dropout=0.2):
        super(MemNN, self).__init__()
        self.hop_count = hop_count
        self.embedding_dim = embedding_dim
        
        # Position encoding
        self.pe = position_encoding(max_sent_len, embedding_dim)
        
        # Embeddings for each hop
        self.A = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim) for _ in range(hop_count + 1)])
        self.C = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim) for _ in range(hop_count + 1)])
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Initialize with small weights
        for emb in self.A: nn.init.normal_(emb.weight, std=0.1)
        for emb in self.C: nn.init.normal_(emb.weight, std=0.1)

    def forward(self, stories, queries):
        """
        stories: (batch, num_sentences, sent_len)
        queries: (batch, query_len)
        """
        device = stories.device
        pe = self.pe.to(device)
        
        # query embedding (initial state)
        u = (self.A[0](queries) * pe[:queries.size(1), :]).sum(dim=1)
        u = self.dropout(u)
        
        all_probs = []
        for k in range(self.hop_count):
            # m_i = sum(A_k(story_i) * PE)
            m = (self.A[k](stories) * pe).sum(dim=2)  # (batch, num_sentences, embedding_dim)
            m = self.dropout(m)
            
            # c_i = sum(C_k(story_i) * PE)
            c = (self.C[k](stories) * pe).sum(dim=2)  # (batch, num_sentences, embedding_dim)
            c = self.dropout(c)
            
            # dot product attention
            probs = F.softmax(torch.bmm(u.unsqueeze(1), m.transpose(1, 2)), dim=-1) # (batch, 1, num_sentences)
            all_probs.append(probs.squeeze(1))
            
            # response vector o = sum(probs * c_i)
            o = torch.bmm(probs, c).squeeze(1) # (batch, embedding_dim)
            
            # update u with residual and layer norm
            u = self.layer_norm(u + o)
            
        # final answer prediction
        logits = F.linear(u, self.C[-1].weight)
        return logits, all_probs
