'''
Basic CNN from scratch to learn the MNIST digits dataset.
HW6 - CS 519
Ezra Huscher
April 2023
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
from torchinfo import summary
from torchviz import make_dot
import time

plt.style.use('ggplot')
transform = transforms.Compose([transforms.ToTensor()])

# Parameters of the neural network
num_epochs = 10
batch_size = 16
num_channels_1 = 8      # output channels for first 2dConv layer
num_channels_2 = 32     # output channels for first 2dConv layer
num_channels_3 = 256    # output channels for first 2dConv layer
num_classes = 10        # 0-9 digits so 10
kernel_val = 3          # size of kernel (square)
torch.manual_seed(1)    # random seed built into PyTorch

# Load up the MNIST dataset (70,000 handwritten digits)
mnist_dataset = datasets.MNIST(root='./data', train=True,transform=transform, download=True)
mnist_valid_dataset = torch.utils.data.Subset(mnist_dataset,torch.arange(10000))
mnist_train_dataset = torch.utils.data.Subset(mnist_dataset,torch.arange(10000, len(mnist_dataset)))
mnist_test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Output stats of Dataset
print('number of items in mnist_dataset:', len(mnist_dataset))
print('number of items in mnist_train_dataset:', len(mnist_train_dataset))
print('number of items in mnist_valid_dataset:', len(mnist_valid_dataset))
print('number of items in mnist_test_dataset:', len(mnist_test_dataset))

# Load training and validation into PyTorch
train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(mnist_valid_dataset, batch_size, shuffle=False)

# Let's build our CNN:
model = torch.nn.Sequential()
model.add_module('conv1', torch.nn.Conv2d(in_channels=1, out_channels=num_channels_1, kernel_size=kernel_val, stride=(1, 1), padding=0))
model.add_module('conv1-2', torch.nn.Conv2d(in_channels=num_channels_1, out_channels=num_channels_1, kernel_size=kernel_val, stride=(1, 1), padding=0))
model.add_module('relu1', torch.nn.ReLU())
model.add_module('pool1', torch.nn.MaxPool2d(2, stride=2))
model.add_module('conv2', torch.nn.Conv2d(in_channels=num_channels_1, out_channels=num_channels_2, kernel_size=kernel_val, stride=(1, 1), padding=1))
model.add_module('conv2-2', torch.nn.Conv2d(in_channels=num_channels_2, out_channels=num_channels_2, kernel_size=kernel_val, stride=(1, 1), padding=0))
model.add_module('relu2', torch.nn.ReLU())
model.add_module('pool2', torch.nn.MaxPool2d(4, stride=4))
model.add_module('conv3', torch.nn.Conv2d(in_channels=num_channels_2, out_channels=num_channels_3, kernel_size=kernel_val, stride=(1, 1), padding=1))
model.add_module('conv3-2', torch.nn.Conv2d(in_channels=num_channels_3, out_channels=num_channels_3, kernel_size=kernel_val, stride=(1, 1), padding=1))
model.add_module('relu3', torch.nn.ReLU())
model.add_module('pool3', torch.nn.MaxPool2d(2, stride=2))
model.add_module('flatten', torch.nn.Flatten())
model.add_module('drop1', torch.nn.Dropout(p=0.5))
model.add_module('fc1', torch.nn.Linear(num_channels_3, num_classes))

'''
# Initial attempt: accuracy: 40%
model.add_module('conv1', torch.nn.Conv2d(in_channels=1,out_channels=4,kernel_size=1,padding='same'))
model.add_module('relu2', torch.nn.ReLU())
model.add_module('pool1', torch.nn.MaxPool2d(2, stride=2))
model.add_module('conv2', torch.nn.Conv2d(in_channels=4,out_channels=2,kernel_size=3,padding='same'))
model.add_module('relu2', torch.nn.ReLU())
model.add_module('pool2', torch.nn.MaxPool2d(4, stride=4))
'''

# Print out summary of each layers
print("Batch size / No. of Channels / Width / Height")
summary(model, input_size=(batch_size, 1, 28, 28))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# Function to train the CNN with PyTorch
def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):

        model.train() # activates Dropout layers during model training
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        
        model.eval() # turns off Dropouts during model evaluation
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0) 
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float() 
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        
        print(f'Epoch {epoch+1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}')
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


tic = time.perf_counter()
hist = train(model, num_epochs, train_dl, valid_dl)
toc = time.perf_counter()
training_time = toc - tic
print("Training time:",training_time)

# Print the final accuracy
pred = model(mnist_test_dataset.data.unsqueeze(1) / 255.)

print("-------------")
print("What it guessed:\nClass the model predicted: ",(torch.argmax(pred, dim=1)))
print("What it actually is:\nClass from the training set: ",(mnist_test_dataset.targets).float())  # Wha

# Calculate and print test accuracy
is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()
print(f'Test accuracy: {is_correct.mean():.4f}')  # is_correct is a tensor

# Save image of CNN diagram
make_dot(pred, params=dict(list(model.named_parameters()))).render("CNN_diagram", format="png")

# ----------------------------
x_arr = np.arange(len(hist[0])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0], '-o', label='Train loss')
ax.plot(x_arr, hist[1], '--<', label='Validation loss')
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2], '-o', label='Train acc.')
ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
file_to_save = "Model_"+str(num_channels_1)+"-"+str(num_channels_2)+"_"+str(kernel_val)+"k_"+str(num_epochs)+"e.png"
plt.savefig(file_to_save, bbox_inches='tight', dpi=200)
plt.show()