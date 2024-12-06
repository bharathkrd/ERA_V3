import pytest
import torch
from mnist_model import TinyMNIST, main

def test_parameter_count():
    model = TinyMNIST()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params < 25000, f"Model has {num_params} parameters, should be less than 25000"

def test_accuracy():
    num_params, accuracy = main()
    assert accuracy >= 95.0, f"Model accuracy is {accuracy}%, should be at least 95%" 