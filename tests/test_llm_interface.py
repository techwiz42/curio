import unittest
import torch
from curious_llm.llm_interface import HuggingFaceLLMInterface
from transformers import AutoModel, AutoTokenizer

class TestHuggingFaceLLMInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AutoModel.from_pretrained("bert-base-uncased")
        cls.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        cls.interface = HuggingFaceLLMInterface(cls.model, cls.tokenizer)

    def test_tokenize(self):
        text = "Hello, world!"
        tokens = self.interface.tokenize(text)
        self.assertIsInstance(tokens, dict)
        self.assertIn('input_ids', tokens)
        self.assertIn('attention_mask', tokens)

    def test_get_embeddings(self):
        text = "Test sentence"
        tokens = self.interface.tokenize(text)
        embeddings = self.interface.get_embeddings(tokens)
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[1], self.model.config.hidden_size)

    def test_get_attention_mask(self):
        input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        attention_mask = self.interface.get_attention_mask(input_ids)
        self.assertTrue(torch.all(attention_mask == torch.tensor([[1, 1, 1, 1, 1]])))

if __name__ == '__main__':
    unittest.main()
