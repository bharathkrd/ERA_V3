import tensorflow as tf
import numpy as np
from mnist_model import create_model, train_model

def test_parameter_count():
    model = create_model()
    total_params = model.count_params()
    assert 15000 < total_params < 25000, f"Model has {total_params} parameters, should be between 15000 and 25000"

def test_model_accuracy():
    model, history = train_model()
    final_accuracy = history.history['accuracy'][-1]
    assert final_accuracy >= 0.95, f"Model accuracy is {final_accuracy}, should be >= 0.95"

if __name__ == "__main__":
    test_parameter_count()
    test_model_accuracy() 