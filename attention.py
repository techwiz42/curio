import torch
import torch.nn as nn

class EntropyBasedAttention(nn.Module):
    def __init__(self, attention_layer):
        super().__init__()
        self.base_attention = attention_layer
    
    def forward(self, query, key, value, attention_mask=None, head_mask=None, entropy_weights=None):
        attention_scores = self.base_attention(query, key, value, attention_mask, head_mask)
        
        if entropy_weights is not None:
            attention_scores = attention_scores * entropy_weights.unsqueeze(1).unsqueeze(1)
        
        return attention_scores
