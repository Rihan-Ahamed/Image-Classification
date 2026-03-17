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

### Name: RIHAN AHAMED S
### Register Number:212224040276
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

        print('Name: RIHAN AHAMED.S')
        print('Register Number:21224040276')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch
<img width="353" height="79" alt="image" src="https://github.com/user-attachments/assets/842109b1-73b9-401d-96f0-607dbccd91d7" />



### Confusion Matrix
<img width="922" height="781" alt="image" src="https://github.com/user-attachments/assets/e91316a1-0df3-4ddb-a61d-0b03ed9bf8e9" />


### Classification Report

<img width="715" height="444" alt="image" src="https://github.com/user-attachments/assets/d43f47e4-4e4f-4295-816f-a3c32055a2ad" />




### New Sample Data Prediction
<img width="638" height="591" alt="image" src="https://github.com/user-attachments/assets/4e3cae6f-a31b-4ab7-895d-9eb9bb15cf67" />



## RESULT
In this experiment, a Convolutional Deep Neural Network (CNN) was successfully implemented.

