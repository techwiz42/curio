import torch
import torch.nn as nn

class CuriosityQueryGenerator(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.query_gen = nn.Linear(hidden_size, hidden_size)
        self.num_attention_heads = num_attention_heads
    
    def forward(self, hidden_states, high_entropy_regions):
        batch_size, seq_length, _ = hidden_states.size()
        queries = []
        
        for region in high_entropy_regions:
            region_hidden = hidden_states[:, region, :]
            region_query = self.query_gen(region_hidden.mean(dim=1))
            queries.append(region_query)
        
        if queries:
            queries = torch.stack(queries, dim=1)
            queries = queries.view(batch_size, -1, self.num_attention_heads, hidden_size // self.num_attention_heads)
            queries = queries.permute(0, 2, 1, 3)
        else:
            queries = torch.empty(batch_size, self.num_attention_heads, 0, hidden_size // self.num_attention_heads, device=hidden_states.device)
        
        return queries
