import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps   # ← added

# Define the CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Input: 1x28x28, Output: 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64x28x28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                    # Output: 64x14x14
        self.fc1 = nn.Linear(64 * 14 * 14, 128)                             # Flatten -> 128 neurons
        self.fc2 = nn.Linear(128, 10)                                       # Output: 10 classes (digits 0-9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load and preprocess data


model     = SimpleCNN()


image_path = "C:\\Users\\alexa\\Pictures\\conv_image.png"

pil_image = Image.open(image_path).convert("L")
invImage  = ImageOps.invert(pil_image)

img_tensor = transforms.ToTensor()(invImage)
img_tensor = transforms.Normalize((0.5,), (0.5,))(img_tensor)


img_tensor = img_tensor.unsqueeze(0)


model.load_state_dict(torch.load('mnist_cnn.pth', map_location='cpu'))
model.eval()
with torch.no_grad():
    out  = model(img_tensor)
    _, pred = torch.max(out, 1)
    print(f'Predicted digit: {pred.item()}')
