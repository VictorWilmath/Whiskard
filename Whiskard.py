import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

catdataset = datasets.ImageFolder(root=r'C:\Users\victo\OneDrive\Desktop\whiskard\cat data set', transform=transform)

train_size = int(0.8 * len(catdataset))
test_size = len(catdataset) - train_size
train_dataset, test_dataset = random_split(catdataset, [train_size, test_size])

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(catdataset.classes)

#neural network

class Whiskard(nn.Module):
    def __init__(self, input_channels, output_features, hidden_units=128):
        super().__init__()
        self.convolutional_layer_stack = nn.Sequential(

            #conv2d done to do color detection   
            #ReLU is chosen becuase its the only nonlinear transformation im comfortable with lol

            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #linear layer

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=64 * 74 * 74, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    
    #forward pass

    def forward(self, x):
        x = self.convolutional_layer_stack(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer_stack(x)
        return x

num_classes = len(catdataset.classes)
num_channels = 3

whiskard = Whiskard(input_channels=num_channels, output_features=num_classes, hidden_units=128).to(device)

lossfn = nn.CrossEntropyLoss()

optfn = torch.optim.SGD(whiskard.parameters(), lr=0.1) #SGD chosen because I determined it to be the best based on my own testing

epochs = 100 #100 epochs chosen because any more is redundant

#training loop

for epoch in range(epochs):
    whiskard.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) 
        def closure():
            optfn.zero_grad()
            outputs = whiskard(images)
            loss = lossfn(outputs, labels)
            loss.backward()
            return loss

        optfn.step(closure)
        running_loss += closure().item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')



#testing loop
whiskard.eval()  
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = whiskard(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')

#saving

output_dir = r'C:\Users\victo\OneDrive\Desktop\whiskard\Whiskard\whiskardbuilds'
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, 'whiskard_model.pt')
torch.save(whiskard.state_dict(), model_path)
print(f"Model saved to: {model_path}")