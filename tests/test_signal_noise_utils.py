import unittest
import numpy as np
from curious_llm.signal_noise_utils import multi_scale_entropy, semantic_clustering, frequency_analysis, signal_noise_distinction
from curious_llm.llm_interface import HuggingFaceLLMInterface
from transformers import AutoModel, AutoTokenizer

class MockLLMInterface:
    def tokenize(self, text):
        return {'input_ids': torch.tensor([[1, 2, 3, 4, 5]])}

    def get_embeddings(self, tokens):
        return np.random.rand(5, 10)  # 5 tokens, 10-dimensional embeddings

class TestSignalNoiseUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_interface = MockLLMInterface()
        cls.real_model = AutoModel.from_pretrained("bert-base-uncased")
        cls.real_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        cls.real_interface = HuggingFaceLLMInterface(cls.real_model, cls.real_tokenizer)

    def test_multi_scale_entropy(self):
        text = "This is a test sentence."
        entropy_profiles = multi_scale_entropy(text, self.mock_interface, scales=[1, 2])
        self.assertEqual(len(entropy_profiles), 2)
        self.assertIsInstance(entropy_profiles[0], list)

    def test_semantic_clustering(self):
        embeddings = np.random.rand(10, 5)  # 10 tokens, 5-dimensional embeddings
        high_entropy_regions = [0, 2, 5, 7]
        cluster_labels = semantic_clustering(high_entropy_regions, embeddings)
        self.assertEqual(len(cluster_labels), len(high_entropy_regions))
        self.assertTrue(all(label in [0, 1] for label in cluster_labels))

    def test_frequency_analysis(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        freq_analysis = frequency_analysis(text)
        self.assertIsInstance(freq_analysis, np.ndarray)
        self.assertEqual(len(freq_analysis), len(text))

    def test_signal_noise_distinction(self):
        text = "This is a complex sentence with potential high-entropy regions."
        signal_regions = signal_noise_distinction(text, self.real_interface)
        self.assertIsInstance(signal_regions, list)
        self.assertTrue(all(isinstance(region, int) for region in signal_regions))

if __name__ == '__main__':
    unittest.main()
