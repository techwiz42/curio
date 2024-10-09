from abc import ABC, abstractmethod
import torch

class LLMInterface(ABC):
    @abstractmethod
    def forward(self, input_ids, attention_mask=None, **kwargs):
        pass

    @abstractmethod
    def get_attention_mask(self, input_ids):
        pass

class HuggingFaceLLMInterface(LLMInterface):
    def __init__(self, model):
        self.model = model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, **kwargs)

    def get_attention_mask(self, input_ids):
        return torch.ones_like(input_ids)
