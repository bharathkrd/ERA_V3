import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Constants
MODEL_PATH = 'mnist_model.pth'

class TinyMNIST(nn.Module):
    def __init__(self):
        super(TinyMNIST, self).__init__()
        # First conv block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second conv block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Classifier
        self.fc = nn.Linear(16 * 7 * 7, 10)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 14x14x8
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 7x7x16
        x = self.dropout(x)
        
        # Classifier
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def get_transforms():
    """Return the transforms used for MNIST"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def train_model():
    """Train and return a new model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = TinyMNIST().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = F.nll_loss(output, target)
        
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}]: '
                  f'Loss: {running_loss/(batch_idx+1):.4f} '
                  f'Accuracy: {100. * correct / total:.2f}%')
    
    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)
    return model

def load_model():
    """Load a trained model"""
    model = TinyMNIST()
    model.load_state_dict(torch.load(MODEL_PATH))
    return model

def main():
    model = train_model()
    
    # Print parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"Number of parameters: {num_params:,}")
    
    # Test accuracy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = datasets.MNIST('./data', train=False, transform=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
    
    accuracy = 100. * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main() 