# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

## STEP 1: Data Preparation

Import necessary libraries

Load MNIST dataset

Apply transformations (ToTensor, normalization)

Create DataLoader for training and testing

## STEP 2: Model Construction

Define CNN class using nn.Module

Add convolution, pooling, and fully connected layers

Define forward propagation

## STEP 3: Training and Evaluation

Define loss function: CrossEntropyLoss()

Define optimizer: Adam()

Train model for multiple epochs

Calculate training loss

Evaluate using confusion matrix and classification report

Test with new sample image


## PROGRAM

### Name: MAHALINGA JEYANTH V
### Register Number:212224220057
```
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

```python
def train_model(model, train_loader, num_epochs=3):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name:MAHALINGA JEYANTH.V')
        print('Register Number:21224220057')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch
<img width="373" height="86" alt="Screenshot 2026-02-11 165635" src="https://github.com/user-attachments/assets/577a8f88-92d9-4d20-9260-9655fc7c2617" />
<img width="361" height="81" alt="Screenshot 2026-02-11 165650" src="https://github.com/user-attachments/assets/7f3b3666-36b1-446d-ab1d-ced6edf642ed" />
<img width="363" height="90" alt="Screenshot 2026-02-11 165821" src="https://github.com/user-attachments/assets/1be9b854-dc4f-4a77-9bc0-45a6ca267385" />


### Confusion Matrix

<img width="828" height="633" alt="Screenshot 2026-02-11 165900" src="https://github.com/user-attachments/assets/3581f2a8-0f44-471f-9828-65cd8ed2bddd" />


### Classification Report

<img width="492" height="329" alt="Screenshot 2026-02-11 165916" src="https://github.com/user-attachments/assets/5c540c1e-b783-4c4e-95b1-0dcb75d88464" />



### New Sample Data Prediction

<img width="610" height="452" alt="Screenshot 2026-02-11 165931" src="https://github.com/user-attachments/assets/c7e3c018-4124-491e-a65f-ae648037da2d" />


## RESULT
In this experiment, a Convolutional Deep Neural Network (CNN) was successfully implemented.

