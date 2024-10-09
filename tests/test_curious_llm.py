import unittest
import torch
from curious_llm.model import CuriousLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestCuriousLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        cls.curious_model = CuriousLLM(cls.base_model)

    def test_model_initialization(self):
        self.assertIsInstance(self.curious_model, CuriousLLM)
        self.assertIsInstance(self.curious_model.llm, HuggingFaceLLMInterface)

    def test_forward_pass(self):
        input_text = "This is a test input."
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.curious_model(input_ids)
        self.assertIn('logits', outputs)
        self.assertEqual(outputs.logits.shape[1], input_ids.shape[1])

    def test_curiosity_query_generation(self):
        input_text = "This is a test input with potential high entropy regions."
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.curious_model(input_ids)
        # Assuming we've added a way to access curiosity queries in the output
        self.assertIn('curiosity_queries', outputs)
        self.assertIsInstance(outputs.curiosity_queries, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
