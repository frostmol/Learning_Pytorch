import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class GeometricFigureDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.classes = ['Circle', 'Heptagon', 'Hexagon', 'Nonagon', 'Octagon', 'Pentagon', 'Square', 'Star', 'Triangle']
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []

        print(f"Image folder: {image_folder}")
        print(f"Contents: {os.listdir(image_folder)}")

        for filename in os.listdir(image_folder):
            if filename.endswith('.png'):  # Assuming all images are .png files
                img_path = os.path.join(image_folder, filename)
                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')  # Ensure RGB mode
                    img = img.resize((200, 200))
                    
                    # Extract the class label from the filename
                    for cls in self.classes:
                        if filename.lower().startswith(cls.lower()):
                            label = self.class_to_index[cls]
                            break
                    else:
                        label = -1  # Unknown class
                    
                    self.images.append(img)
                    self.labels.append(label)
                    #print(f"Loaded image: {img_path}, Label: {self.classes[label]}")
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")

        print(f"Total images loaded: {len(self.images)}")
        print(f"Labels: {set(self.labels)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define dataset path
dataset_path = "dataset/train"

# Create dataset
dataset = GeometricFigureDataset(dataset_path, transform=transform)

# Create data loader
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class GeometricFigureCNN(nn.Module):
    def __init__(self):
        super(GeometricFigureCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 25 * 25, 512)
        self.fc2 = nn.Linear(512, 9)  # 9 classes
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        #print(x.shape)
        x = x.view(-1, 128 * 25 * 25)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = GeometricFigureCNN().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 20
best_accuracy = 0

for epoch in range(num_epochs):
    print('AAAAAAAAAAA Epoch')
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    model.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    
    print(f'Epoch [{epoch+1}/{num_epochs}] Accuracy: {accuracy:.2f}%')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')

print(f"Best training accuracy: {best_accuracy:.2f}%")
