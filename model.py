import torch
import torch.nn as nn
from .llm_interface import LLMInterface, HuggingFaceLLMInterface
from .entropy import calculate_entropy, identify_high_entropy_regions
from .attention import EntropyBasedAttention
from .query_generator import CuriosityQueryGenerator

class CuriousLLM(nn.Module):
    def __init__(self, base_model, entropy_threshold=0.5):
        super().__init__()
        if isinstance(base_model, LLMInterface):
            self.llm = base_model
        else:
            self.llm = HuggingFaceLLMInterface(base_model)
        
        self.entropy_threshold = entropy_threshold
        self.curiosity_query_gen = CuriosityQueryGenerator(base_model.config.hidden_size, base_model.config.num_attention_heads)
        
        # Modify attention layers
        if hasattr(base_model, 'transformer'):
            for layer in base_model.transformer.h:
                layer.attn = EntropyBasedAttention(layer.attn)
        elif hasattr(base_model, 'model'):
            for layer in base_model.model.layers:
                layer.self_attn = EntropyBasedAttention(layer.self_attn)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Initial forward pass
        outputs = self.llm.forward(input_ids, attention_mask=attention_mask, **kwargs)
        
        # Calculate entropy
        logits = outputs.logits
        entropy = calculate_entropy(logits)
        
        # Identify high entropy regions
        high_entropy_regions = identify_high_entropy_regions(entropy, self.entropy_threshold)
        
        # Generate curiosity queries
        hidden_states = outputs.hidden_states[-1]
        curiosity_queries = self.curiosity_query_gen(hidden_states, high_entropy_regions)
        
        # Use curiosity queries for additional processing
        # This part would depend on how you want to use the curiosity queries
        # For example, you could use them to generate additional outputs or modify the existing outputs
        
        return outputs
