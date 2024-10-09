from abc import ABC, abstractmethod
import torch

class LLMInterface(ABC):
    @abstractmethod
    def forward(self, input_ids, attention_mask=None, **kwargs):
        pass

    @abstractmethod
    def get_attention_mask(self, input_ids):
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def get_embeddings(self, tokens):
        pass

class HuggingFaceLLMInterface(LLMInterface):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, **kwargs)

    def get_attention_mask(self, input_ids):
        return torch.ones_like(input_ids)

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt")

    def get_embeddings(self, tokens):
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.squeeze(0).numpy()
