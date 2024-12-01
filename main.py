import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision import transforms
import numpy as np

# Load the trained model
class GeometricFigureCNN(torch.nn.Module):
    def __init__(self):
        super(GeometricFigureCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 25 * 25, 512)
        self.fc2 = torch.nn.Linear(512, 9)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 25 * 25)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GeometricFigureCNN().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Define classes
classes = ['Circle', 'Heptagon', 'Hexagon', 'Nonagon', 'Octagon', 'Pentagon', 'Square', 'Star', 'Triangle']

# Define transformations
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Drawing application
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=400, height=400, bg="white")
        self.canvas.pack()

        self.image = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.predict_button = tk.Button(root, text="Predict Shape", command=self.predict_shape)
        self.predict_button.pack()

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.image)

    def predict_shape(self):
        # Process the drawn image
        image = self.image
        image = ImageOps.invert(image)  # Invert colors (background should be black for the model)
        image = image.resize((200, 200))  # Resize to model's input size

        # Transform image
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            shape = classes[predicted.item()]

        # Show result
        result_window = tk.Toplevel(self.root)
        result_label = tk.Label(result_window, text=f"Predicted Shape: {shape}", font=("Arial", 16))
        result_label.pack()

# Initialize the Tkinter app
root = tk.Tk()
root.title("Draw a Shape")
app = DrawingApp(root)
root.mainloop()
