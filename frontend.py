import sys
import torch
from torch import nn
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from torchvision import transforms
from PIL import Image

class Whiskard(nn.Module):
    def __init__(self, input_channels, output_features, hidden_units=128):
        super().__init__()
        self.convolutional_layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=64 * 74 * 74, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    
    def forward(self, x):
        x = self.convolutional_layer_stack(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer_stack(x)
        return x
model = Whiskard(input_channels=3, output_features=3, hidden_units=128)
model.load_state_dict(torch.load(r'C:\Users\victo\OneDrive\Desktop\whiskard\Whiskard\whiskardbuilds\whiskard_model.pt'))
model.eval()
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Whiskard(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle('Whiskard V 1.0')
        self.setGeometry(100, 100, 400, 200)
        layout = QVBoxLayout()
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.upload_button = QPushButton('Upload Image', self)
        self.upload_button.clicked.connect(self.getpicture)
        layout.addWidget(self.upload_button)
        self.result_label = QLabel(self)
        layout.addWidget(self.result_label)
        self.setLayout(layout)
   
    def getpicture(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaledToWidth(300))
            cat_name, confidence = self.predictCat(file_name)
            self.result_label.setText(f'Predicted Cat: {cat_name} (Confidence: {confidence:.2f})')

    def predictCat(self, image_path):
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, predicted_class = torch.max(probabilities, dim=0)
            cats = ['beans', 'pips', 'sammy']
            mostlikelycat = cats[predicted_class.item()]
        return mostlikelycat, confidence.item()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Whiskard()
    window.show()
    sys.exit(app.exec_())