import re
import torch
from torch.utils.data import Dataset
from collections import Counter

def tokenize(sent):
    """
    Split a sentence into tokens including punctuation.
    """
    return [x.strip() for x in re.split(r'(\W+)', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):
    """
    Parse stories in the bAbI format.
    """
    data = []
    story = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            # a is a single word answer in bAbI
            substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def get_stories(f):
    """
    Read stories from a file.
    """
    with open(f, 'r') as f:
        return parse_stories(f.readlines())

class BabiDataset(Dataset):
    def __init__(self, data, word2idx, max_story_len, max_query_len):
        self.data = data
        self.word2idx = word2idx
        self.max_story_len = max_story_len
        self.max_query_len = max_query_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        story, query, answer = self.data[idx]
        
        # Flatten and pad story
        # In MemNN, each sentence is often treated separately. 
        # But for simple version, we can treat it as a list of sentences.
        
        s_vec = []
        for sent in story:
            s_vec.append([self.word2idx.get(w, 0) for w in sent])
        
        q_vec = [self.word2idx.get(w, 0) for w in query]
        a_vec = self.word2idx.get(answer, 0)
        
        return s_vec, q_vec, a_vec
