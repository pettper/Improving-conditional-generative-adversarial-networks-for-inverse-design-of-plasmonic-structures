import unittest
from src.utils import contractive_penalty, decoder_penalty, contractive_penalty_v2, decoder_penalty_v2
import torch
import numpy as np


class TestPenaltyTerms(unittest.TestCase):

    def test_contractive_penalty(self):

        class SimpleTestClass:

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def encode(self, x):
                return x

        B = 16
        D = 40

        model = SimpleTestClass()
        x = torch.ones(B, D, requires_grad=True)
        p = contractive_penalty(model, x)

        self.assertEqual(p.shape[0], B)
        self.assertTrue(p.equal(torch.Tensor([np.sqrt(40) for _ in range(B)])))

    def test_decoder_penalty(self):

        B = 16
        D = 40

        class SimpleTestClass:

            def __init__(self):
                pass

            def decode(self, x):
                return x

            def encode(self, x):
                return x

        model = SimpleTestClass()
        x = torch.ones(B, D, requires_grad=True)
        p = decoder_penalty(model, x)

        self.assertEqual(p.shape[0], B)
        self.assertTrue(p.equal(torch.Tensor([np.sqrt(40) for _ in range(B)])))

    def test_contractive_penalty_v2(self):

        class SimpleTestClass:

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def encode(self, x):
                return x

        B = 16
        D = 40

        model = SimpleTestClass()
        x = torch.ones(B, D, requires_grad=True)
        p = contractive_penalty_v2(model, x, torch.device('cpu'))

        self.assertEqual(p.shape[0], B)
        self.assertTrue(p.equal(torch.Tensor([np.sqrt(40) for _ in range(B)])))

    def test_decoder_penalty_v2(self):

        B = 16
        D = 40

        class SimpleTestClass:

            def __init__(self):
                pass

            def decode(self, x):
                return x

            def encode(self, x):
                return x

        model = SimpleTestClass()
        x = torch.ones(B, D, requires_grad=True)
        p = decoder_penalty_v2(model, x, torch.device('cpu'))

        self.assertEqual(p.shape[0], B)
        self.assertTrue(p.equal(torch.Tensor([np.sqrt(40) for _ in range(B)])))


if __name__ == '__main__':
    unittest.main()
