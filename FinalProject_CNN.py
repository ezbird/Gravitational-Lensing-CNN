import pandas as pd
import numpy as np
import math
import time
from skimage.io import imread
from skimage.transform import rotate, resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Flatten
from torch.optim import Adam, lr_scheduler
from torchinfo import summary
import optuna
from optuna.visualization.matplotlib import plot_contour


print("-----------------------------------------------------------------")
print("Gravitational Lensing Detection CNN for the COSMOS2020 Catalog")
print("-----------------------------------------------------------------")

# Parameters for the CNN
start_channels     = 1     # In an RGB image, we'd use 3... but as our lenses our grayscale, let's use 1
image_width        = 84
image_height       = 84
batch_size         = 32     # batch size of the model
n_epochs           = 10     # number of epochs to train the model
learning_rate      = 0.001  # optimizer learning rate, i.e. 0.001
kernel_conv_size   = 2
kernel_pool_size   = 2
stride_conv_size   = 1      # cannot change these without altering linear layers
stride_pool_size   = 2      # cannot change these without altering linear layers
num_classes        = 2      # Is a lens or is not
augment_data       = False  # This flips and rotates training images to increase the training set. 
                            # Process requires a lot of memory though!
                            # Faster to just make the augmented images beforehand.


tic = time.perf_counter()  # start training timer

# Get the CSV file that denotes which class 
# it is (lens or not), i.e. 1 or 0, correlating to each galaxy image.
# To speed up testing: "skiprows=lambda i: i % 2" to reduce dataset.
train = pd.read_csv('galaxy_labels.csv', skiprows=lambda i: i % 4)  #, skiprows=lambda i: i % 2

# Load in and resize all the galaxy images to 200x200.
# I don't know if this is the optimal size or not?
print("Start loading dataset...")
train_img = []
train_img_values = []
for img_name in train['image_names']:
    image_path = 'galaxy_images/' + img_name 
    img = imread(image_path)
    img = img/255
    img = resize(img, output_shape=(start_channels, image_width, image_height),mode='constant', anti_aliasing=True)
    train_img.append(img)


train_x = np.array(train_img)

# Create a variable that holds label 1 (lens) or 0 (not a lens)
train_y = train['lens_or_not'].values

# Let's split the data set into groups of 80% training and 20% test.
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=13, stratify=train_y)
print("Images in Training set : ",train_x.shape[0],"  Images in Testing set : ",test_x.shape[0])

toc = time.perf_counter()
print(f"Data loading took {toc - tic:0.4f} seconds")

# ----------------------------------------------------------
# Image Augmentation (optional; this works well but is slow)
# ----------------------------------------------------------
if (augment_data):

    final_train_x = []
    final_train_y = []

    print("train_x.shape[0]",train_x.shape[0])
    for i in range(math.ceil(train_x.shape[0])):
        final_train_x.append(train_x[i])
        final_train_x.append(rotate(train_x[i], angle=45, mode = 'wrap'))
        final_train_x.append(np.fliplr(train_x[i]))
        for j in range(3):
            final_train_y.append(train_y[i])

    train_x = np.array(final_train_x)
    train_y = np.array(final_train_y)
    print("Number of images (After Image Augmentation) in Training set : ",train_x.shape[0],"  Number of images in Testing set : ",test_x.shape[0])
# ----------------------------------------------------------


# converting training images and its labels into torch format
train_x = train_x.reshape(train_x.shape[0], start_channels, image_width, image_height) # the first number here is the number of training images
train_x  = torch.from_numpy(train_x)
train_x = train_x.float()
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y)

# converting test images and its labels into torch format
test_x = test_x.reshape(test_x.shape[0], start_channels, image_width, image_height)
test_x  = torch.from_numpy(test_x)
test_x = test_x.float()
test_y = test_y.astype(int)
test_y = torch.from_numpy(test_y)

# ----------------------------------------------------------
# Create the CNN structure to identify gravitational lenses
# ----------------------------------------------------------
class LensDetectCNN(Module):   
    def __init__(self):
        super(LensDetectCNN, self).__init__()

        self.cnn_layers = Sequential(
            #Conv2d(3, 32, kernel_size=kernel_conv_size, stride=stride_conv_size, padding=1),
            #ReLU(inplace=True),
            #BatchNorm2d(32),
            #MaxPool2d(kernel_size=kernel_pool_size, stride=stride_pool_size),
            #Dropout(p=0.25),
            Conv2d(start_channels, 32, kernel_size=kernel_conv_size, stride=stride_conv_size, padding=1),
            ReLU(inplace=True),
            #BatchNorm2d(32),
            MaxPool2d(kernel_size=kernel_pool_size, stride=stride_pool_size),
            #Dropout(p=0.25),
            Conv2d(32, 64, kernel_size=kernel_conv_size, stride=stride_conv_size, padding=1),
            ReLU(inplace=True),
            #BatchNorm2d(128),
            MaxPool2d(kernel_size=kernel_pool_size, stride=stride_pool_size),
            #Dropout(p=0.25),
            Conv2d(64, 1024, kernel_size=kernel_conv_size, stride=stride_conv_size, padding=1),
            ReLU(inplace=True),
            #BatchNorm2d(128),
            MaxPool2d(kernel_size=kernel_pool_size, stride=stride_pool_size),
            Flatten(),
            #Dropout(p=0.25),
        )

        self.linear_layers = Sequential(
            Linear(1024 * 11 * 11, 10),
            ReLU(inplace=True),
            #Dropout(),
            #Linear(512, 256),
            #ReLU(inplace=True),
            #Dropout(),
            #Linear(512,10),
            #ReLU(inplace=True),
            #Dropout(),
            Linear(10,2)
        )

    # Define the whole forward pass here.
    # This generates predictions based on the current model parameters
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
# ----------------------------------------------------------


# ----------------------------------------------------------
# Optune will aid the optimization step by trying many 
# different combinations of learning rate and weight decay
# to get the best results. It does take time!
# ----------------------------------------------------------
print("Start optimizing model...")
class optimize_cnn(object):
    """
    Optimize the params
    """
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def __call__(self, trial):
        #Only optimizing two parameters in this example, lr and decay
        learning_rate = trial.suggest_float('lr', 1e-6, 1e-3, step=5e-6) 
        weight_decay = trial.suggest_float('decay', 0.0, 0.1, step=1e-4)

        print("Start building model...")
        tic = time.perf_counter()  # start training timer

        model = LensDetectCNN()
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # experimenting with weight decay
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
        criterion = CrossEntropyLoss()

        # Print out summary of each layers from torchinfo
        print("Batch size / No. of Channels / Width / Height")
        summary(model, input_size=(batch_size, start_channels, image_width, image_height))

        # Start training with specified number of epochs
        for epoch in range(1, n_epochs+1):

            train_loss = 0.0
            permutation = torch.randperm(self.train_x.size()[0])
            training_loss = []
            for i in range(0,self.train_x.size()[0], batch_size):

                optimizer.zero_grad() # do this first?

                indices = permutation[i:i+batch_size]
                batch_x, batch_y = self.train_x[indices], self.train_y[indices]
                
                prediction_outputs = model(batch_x)
                loss = criterion(prediction_outputs,batch_y)
                loss.backward()
                optimizer.step()

                training_loss.append(loss.item())

            scheduler.step()  # Adjust learning rate if we are not getting anywhere
            training_loss = np.average(training_loss)
            print('epoch: \t', epoch, '\t training loss: \t', training_loss)

        toc = time.perf_counter()   # stop training timer
        print(f"Training took {toc - tic:0.4f} seconds")

        return 1-training_loss #Subtract because the direction='maximize'

def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} and parameters: {}. ".format(
            frozen_trial.number,
            frozen_trial.value,
            frozen_trial.params,
            )
        )


sampler = optuna.samplers.TPESampler(seed=1909)
study = optuna.create_study(direction='maximize', sampler=sampler)
print('Starting hyperparameter optimization, this will take a while...')
objective = optimize_cnn(train_x, train_y) 
study.optimize(objective, n_trials=100, show_progress_bar=True, callbacks=[logging_callback])#, n_jobs=1)
params = study.best_trial.params
print('Optimization complete!')

print("Building final model...")
learning_rate = params['lr']
weight_decay = params['decay']

tic = time.perf_counter()  # start training timer

model = LensDetectCNN()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # experimenting with weight decay
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
criterion = CrossEntropyLoss()

# Print out summary of each layers from torchinfo
print("Batch size / No. of Channels / Width / Height")
summary(model, input_size=(batch_size, start_channels, image_width, image_height))

# Start training with specified number of epochs
for epoch in range(1, n_epochs+1):

    train_loss = 0.0
    permutation = torch.randperm(train_x.size()[0])
    training_loss = []
    for i in range(0,train_x.size()[0], batch_size):

        optimizer.zero_grad() # do this first?

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = train_x[indices], train_y[indices]
        
        prediction_outputs = model(batch_x)
        loss = criterion(prediction_outputs,batch_y)
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())

    scheduler.step()  # Adjust learning rate if we are not getting anywhere
    training_loss = np.average(training_loss)
    print('epoch: \t', epoch, '\t training loss: \t', training_loss)

toc = time.perf_counter()   # stop training timer
print(f"Training took {toc - tic:0.4f} seconds")


print("Start predicting...")
# ----------------------------------------------------------
# Prediction for training set
# ----------------------------------------------------------
prediction = []
target = []
permutation = torch.randperm(train_x.size()[0])

# Okay, good good. Now let's loop back over the training set and see how
# our predictions do.
for i in range(0,train_x.size()[0], batch_size):
    indices = permutation[i:i+batch_size]
    batch_x, batch_y = train_x[indices], train_y[indices]

    output = model(batch_x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.detach().numpy())
    predictions = np.argmax(prob, axis=1)
    prediction.append(predictions)
    target.append(batch_y)
    
# training accuracy
accuracy = []
for i in range(len(prediction)):
    accuracy.append(accuracy_score(target[i].cpu(),prediction[i]))
    
print('training accuracy: \t', np.average(accuracy))


# ----------------------------------------------------------
# check the performance on validation set
# ----------------------------------------------------------
torch.manual_seed(0)
# batch size of the model
batch_size = 64
# prediction for test set
prediction = []
target = []
permutation = torch.randperm(test_x.size()[0])
for i in range(0,test_x.size()[0], batch_size):
    indices = permutation[i:i+batch_size]
    batch_x, batch_y = test_x[indices], test_y[indices]

    output = model(batch_x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.detach().numpy())  #.detach()
    predictions = np.argmax(prob, axis=1)
    prediction.append(predictions)
    target.append(batch_y)
    

# ----------------------------------------------------------
# Test accuracy
# ----------------------------------------------------------
accuracy = []
for i in range(len(prediction)):
    accuracy.append(accuracy_score(target[i].cpu(),prediction[i]))
    
print('Test accuracy: \t', np.average(accuracy))
torch.save(model, 'CNN-model.pth')

print("Finished.")

exit()

# ----------------------------------------------------------
# The below code is used to test the model with a specific image.
# ----------------------------------------------------------
model = torch.load('CNN-model.pth')
image_path = 'galaxy_images/407.png' #Specify the image path
img = imread(image_path)
img = img/255
img = resize(img, output_shape=(start_channels, 224, 224),mode='constant', anti_aliasing=True)
img = img.astype('float32')
img = np.array(img)

img = img.reshape(1, start_channels, 224, 224) 
img = torch.from_numpy(img)
img = img.float()

output = model(img)
softmax = torch.exp(output).cpu()
prob = list(softmax.detach().numpy())
predictions = np.argmax(prob, axis=1)
print(predictions)

