# 🎯 MNIST Classification: The Little Model That Could

![MNIST](https://img.shields.io/github/actions/workflow/status/bharathkrd/ERA_V3/model_tests.yml?style=for-the-badge&logo=pytorch&logoColor=FF00FF&labelColor=000000&color=00FFFF&label=🤖%20NEURAL%20MAGIC)

Welcome to our MNIST classifier, where we prove that size doesn't matter (in neural networks, at least 😉). This tiny but mighty model achieves >95% accuracy with fewer parameters than your average Twitter post!

## 🌟 What's Special About This Model?

Our model is like that friend who's super efficient at their job but still manages to leave work early:
- Runs on a parameter diet (< 25K parameters)
- Gets to 95%+ accuracy faster than you can say "deep learning"
- So lightweight, it probably uses less memory than this README

## 🤖 Model Architecture: The Not-So-Secret Sauce

```python
TinyMNIST(
  # First conv block (because you gotta start somewhere)
  Conv2d(1, 8, kernel=3, padding=1)  # 8 filters, because 7 ate 9
  BatchNorm2d(8)  # Keeping those activations in check
  ReLU  # The only thing we're not minimalistic about
  MaxPool2d(2)  # Downsizing like it's fashion
  
  # Second conv block (the sequel is sometimes better)
  Conv2d(8, 16, kernel=3, padding=1)  # Double the filters, double the fun
  BatchNorm2d(16)  # Still keeping it normalized
  ReLU  # If it ain't broke...
  MaxPool2d(2)  # More downsizing
  Dropout(0.2)  # Randomly ghosting 20% of neurons
  
  # Classifier (the grand finale)
  Linear(16*7*7, 10)  # 10 outputs, because humans have 10 fingers
)
```

## 📊 Performance Metrics (The Humble Brag)
- Parameters: 9,146 (Marie Kondo would be proud)
- Training Accuracy: >95% (in ONE epoch, because ain't nobody got time for more!)
- Test Accuracy: >95% (it actually works! 🎉)

## 🛠️ Project Structure
```
session5/
├── mnist_model.py     # Where the magic happens
├── test_model.py      # Trust, but verify
└── README.md         # You are here! 👋
```

## 📋 Prerequisites
- Python 3.8+ (time to upgrade from that Python 2.7!)
- PyTorch 1.9+ (because we're not cavemen)
- torchvision (for those fancy transformations)
- pytest (because we're professionals™)

## 🚀 Quick Start

1. Clone this bad boy:
```bash
git clone https://github.com/bharathkrd/ERA_V3.git
cd ERA_V3/session5  # Enter the matrix
```

2. Create a virtual environment (because we're not savages):
```bash
python -m venv mnist_env
source mnist_env/bin/activate  # Linux/Mac users
mnist_env\Scripts\activate     # Windows users, we got you too!
```

3. Install dependencies:
```bash
pip install torch torchvision pytest  # The holy trinity
```

4. Train the model:
```bash
python mnist_model.py  # Grab a coffee, but hurry back!
```

5. Run tests:
```bash
pytest test_model.py -v  # Watch those green checkmarks roll in
```

## ✨ Features That Make Us Special
- Efficient architecture (because we believe in working smarter, not harder)
- Batch Normalization (keeping our neurons well-behaved)
- Strategic dropout (teaching our model to be independent)
- Data augmentation (because variety is the spice of life)
- Proper train/test pipeline (we're not barbarians)

## 🧪 Test Cases (Because Trust Issues)
- Parameter count check (making sure we stick to our diet)
- Accuracy validation (proof that we're not just making things up)
- Transform normalization (keeping it real... normalized)
- Dropout behavior (making sure our neurons are properly ghosting)
- Model output shape (because size and shape matter)

## ⚖️ License
MIT (because sharing is caring)

## 👨‍💻 Author
Bharath K R (The guy who made a neural network go on a successful diet)

Remember: In a world of ResNets and Transformers, be a TinyMNIST. Small but mighty! 💪

P.S. If this model were any lighter, it would float! 🎈