import torch
import torch.nn.functional as F

def calculate_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)

def identify_high_entropy_regions(entropy, threshold):
    return (entropy > threshold).nonzero().squeeze(-1)
