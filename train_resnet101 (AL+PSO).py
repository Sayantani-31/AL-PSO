# -*- coding: utf-8 -*-
"""AL_no_batch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xqEZfJzf8VhsGQyKyCFcsk5f5bkJHRnN
"""

# from google.colab import drive
# drive.mount('/content/drive')

# !cp /content/drive/MyDrive/HAM/ham10000_augmented_split_dataset_6k.zip ./ham10000_augmented_split_dataset_6k.zip
# !unzip -q -n ham10000_augmented_split_dataset_6k.zip -d ham10000_augmented_split_dataset_6k

import os
import pandas as pd
import numpy as np
import math
import random
import time

# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

# ------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fitness_type', choices=['least_confidence', 'entropy', 'lowest_margin', 'highest_margin'])
# parser.add_argument('-w', '--weight_decay', type=float)
parser.add_argument('-s', '--selection_probability', type=float)
parser.add_argument('-b', '--batch_size', type=int, default=8)


args = parser.parse_args()

#-------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def set_debug_apis(state):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)

set_debug_apis(False)

#-------------------------------------------------------------------------------------------------
"""# **Dataset**"""

dataset_path = "./dataset_4827"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")
val_path = os.path.join(dataset_path, "val")

IMAGE_SIZE = (255, 255)
# BATCH_SIZE = 8

FITNESS_TYPE = args.fitness_type
BATCH_SIZE = args.batch_size
# WEIGHT_DECAY = args.weight_decay
IMAGE_SELECTION_PROB = args.selection_probability

output_dir = f"./save_files/resnet101_{FITNESS_TYPE}_sp{IMAGE_SELECTION_PROB}_bs{BATCH_SIZE}"
os.makedirs(output_dir, exist_ok = True)        # create directory if it doesnt exist

class_names = sorted(os.listdir(train_path))
num_classes = len(class_names)

print("-------------------------------", output_dir.split("/")[-1], "-------------------------------")
print(num_classes, "classes:", class_names)
print("BATCH_SIZE:", BATCH_SIZE)
print("IMAGE_SELECTION_PROB:", IMAGE_SELECTION_PROB)

# print number of images in train and test folder for each label
print("Train images:")
for label in class_names:
    print(label, ":", len(os.listdir(os.path.join(train_path, label))))

print("\nVal images:")
for label in class_names:
    print(label, ":", len(os.listdir(os.path.join(val_path, label))))

print("\nTest images:")
for label in class_names:
    print(label, ":", len(os.listdir(os.path.join(test_path, label))))


#-------------------------------------------------------------------------------------------------
# create a 'metadata' variable that will be a python dictionary,
# key will be image_id (name of the image file)
# and value will include the image's path and label
# we will use this metadata dictionary to find the label of any image later
# this will work as the oracle

metadata = {}

for label in class_names:
    files = os.listdir(os.path.join(train_path, label))
    for filename in files:
        image_id = filename[:-4]    # remove the .jpg extension from filename
        metadata[image_id] = {"path": os.path.join(train_path, label, filename), "label": label}

# create a dictionary to create a numerical representation of each label
label2idx = {class_names[i]:i for i in range(num_classes)}
print(label2idx)


#-------------------------------------------------------------------------------------------------
X = list(metadata.keys())                   # list of all image names
y = [metadata[x]["label"] for x in X]       # list of the image labels

# split train data into 10% labeled data and 90% unlabeled data
X_unlabeled, X_labeled, _, y_labeled = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# call the unlabeled data as 'unlabeled_pool'
# which is just an array of image_ids (image filenames)
unlabeled_pool = X_unlabeled

print("\nDataset lengths:")
print("unlabeled :", len(unlabeled_pool))
print("labeled :", len(X_labeled))
# print("validation :", len(X_val))


#-------------------------------------------------------------------------------------------------
class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image_id = self.X[idx]
        image_path = metadata[image_id]["path"]
        image = torchvision.io.read_image(image_path)
        image = torchvision.transforms.functional.resize(image, IMAGE_SIZE, antialias=True)
        # image = image/255   # normalize the values between 0 to 1

        label_idx = label2idx[self.y[idx]]

        return image, label_idx

    # function to add selected images to labeled dataset
    def add_data(self, X):
        y = [metadata[image_id]["label"] for image_id in X]     # get labels for all the unlabeled images from the metadata dictionary
        self.X += X
        self.y += y

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image_id = self.X[idx]
        image_path = metadata[image_id]["path"]
        image = torchvision.io.read_image(image_path)
        image = torchvision.transforms.functional.resize(image, IMAGE_SIZE, antialias=True)
        # image = image/255   # normalize the values between 0 to 1

        return image, image_id

labeled_dataset = LabeledDataset(X_labeled, y_labeled)
print("Labeled dataset size:", len(labeled_dataset))

train_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print("Train loader size:", len(train_loader))

# val_dataset = LabeledDataset(X_val, y_val)
# print("val dataset size:", len(val_dataset))

# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# print("val loader size:", len(val_loader))

val_dataset = torchvision.datasets.ImageFolder(
    val_path,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMAGE_SIZE, antialias=True),
        torchvision.transforms.ToTensor()
    ])
)
print("val dataset size:", len(val_dataset))

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print("val loader size:", len(val_loader))

#-------------------------------------------------------------------------------------------------
"""# **CNN model**"""

weights = torchvision.models.ResNet101_Weights.DEFAULT          # pretrained weights for model
model = torchvision.models.resnet101(weights=weights)           # define the model with pretrained weights

print(model)

# get the transformations required for
# preprocessing images before sending to pretrained model
pretrained_preprocess = weights.transforms()

# freeze model pretrained weights
for param in model.parameters():
    param.requires_grad = False

# replace last layer
num_ftrs = model.fc.in_features
print("num_ftrs: ", num_ftrs)

model.fc = nn.Sequential(
# model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 128),
    nn.ReLU(True),
    nn.Dropout(p=0.5),
    nn.Linear(128, num_classes),
    nn.Softmax(dim = -1)
)

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-9,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

def calc_acc(label, pred):
    pred = torch.argmax(pred, axis=-1)
    match = label == pred
    return match.sum()/len(pred)


#-------------------------------------------------------------------------------------------------
for x, y in train_loader:
    x = x.to(device)
    x = pretrained_preprocess(x)
    y = y.to(device)

    print("x:", x.shape, x.dtype)
    print("y:", y.shape, y.dtype)
    break

outputs = model(x)

print("output shape:", outputs.shape, outputs.dtype)
print("acc:", calc_acc(y, outputs))

del x, y, outputs


#-------------------------------------------------------------------------------------------------
def train_epoch(train_dl, epoch, num_epochs):

    n_total_steps = len(train_dl)
    print_n_steps = 50
    total_train_loss = 0
    total_train_acc = 0
    steps = 0


    start_time = time.time()
    model.train()
    for x, y in train_dl:

        x = x.to(device)
        x = pretrained_preprocess(x)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True) # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

        outputs = model(x)
        del x   # delete input from memory to save space

        # Compute the loss and its gradients
        loss = loss_fn(outputs, y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Compute masked accuracy
        acc = calc_acc(y, outputs)

        total_train_loss += loss
        total_train_acc += acc

        steps+=1
        if steps%print_n_steps == 0 or steps==1 or steps==n_total_steps:
            elapsed_time = int(((time.time() - start_time)*1000)/steps)

            acc_number = acc.detach().item()
            avg_loss = total_train_loss.detach().item()/steps
            avg_acc = total_train_acc.detach().item()/steps

            print(f"Epoch {epoch+1:2d}/{num_epochs} [{steps:4d}/{n_total_steps:4d}] {elapsed_time:3d}ms/step  loss: {avg_loss:.4f}  acc: {avg_acc:.4f}  [{acc_number:.4f}]")


    avg_loss = total_train_loss.detach().item()/n_total_steps
    avg_acc = total_train_acc.detach().item()/n_total_steps

    return avg_loss, avg_acc


#-------------------------------------------------------------------------------------------------
def val_epoch(val_dl):

    n_val_steps = len(val_dl)
    total_val_loss = 0
    total_val_acc = 0

    model.eval()
    with torch.no_grad():
        outputs = None
        for x, y in val_dl:
            x = x.to(device)
            x = pretrained_preprocess(x)
            y = y.to(device)

            outputs = model(x)
            del x   # delete input from memory to save space

            # Compute the loss and accuracy
            val_loss = loss_fn(outputs, y)

            val_acc = calc_acc(y, outputs)

            total_val_loss += val_loss
            total_val_acc  += val_acc

    avg_loss = total_val_loss.detach().item() / n_val_steps
    avg_acc  = total_val_acc.detach().item()  / n_val_steps

    return avg_loss, avg_acc


#-------------------------------------------------------------------------------------------------
def train_model(num_epochs, output_dir=None, save_weights=False, patience=-1):

    checkpoint_filename = None
    best_checkpoint_filename = None
    logfilepath = None

    if save_weights:
        checkpoint_filename = "model_checkpoint.pth"
        best_checkpoint_filename = "model_best_checkpoint.pth"
        logfilepath = os.path.join(output_dir, "logs.csv")

        if not os.path.isfile(logfilepath):
            with open(logfilepath, "a") as logfile:
                logfile.write("epoch,train_loss,train_acc,val_loss,val_acc\n")


    max_val_acc = 0
    min_val_loss = float("inf")
    patience_count = 0


    for epoch in range(0, num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')

        train_loss, train_acc = train_epoch(train_loader, epoch, num_epochs)
        val_loss, val_acc = val_epoch(val_loader)

        scheduler.step(val_loss)    # adjust learning rate

        print(f"Epoch {epoch+1:2d}/{num_epochs}   train_loss: {train_loss:.4f}   train_acc: {train_acc:.4f}      val_loss: {val_loss:.4f}   val_acc: {val_acc:.4f}")

        if save_weights:
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(output_dir, checkpoint_filename)
            )

            # Save best till now
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(output_dir, best_checkpoint_filename)
                )


            with open(logfilepath, "a") as logfile:
                logfile.write(f"{epoch+1},{train_loss},{train_acc},{val_loss},{val_acc}\n")

        # Early Stopping
        if patience>=0:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                patience_count = 0
            else:
                patience_count+=1
                if patience_count>patience:
                    break

    return train_loss, train_acc, val_loss, val_acc


#-------------------------------------------------------------------------------------------------
train_model(num_epochs=10)


#-------------------------------------------------------------------------------------------------
"""# **Active Learning**"""

# Predict probablities for all unlabeled images

prediction_cache = {}   # will store predicted probabilties for each image_id

unlabeled_ds = UnlabeledDataset(unlabeled_pool)
unlabeled_dl = torch.utils.data.DataLoader(unlabeled_ds, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
with torch.no_grad():
    n_steps = len(unlabeled_dl)
    step = 0
    for x, img_id in unlabeled_dl:
        step+=1
        x = x.to(device)
        x = pretrained_preprocess(x)
        outputs = model(x)

        # print(x)
        # print(img_id)
        # print(outputs)
        # break

        for i in range(len(x)):
            prediction_cache[img_id[i]] = outputs[i].cpu()

        if step==1 or step%100==0 or step==n_steps:
            print(f"Calculating probabilites [{step:4d}/{n_steps}]")



def get_prediction_probabilites(imageid_list):
    # return torch.from_numpy(np.array([prediction_cache[img_id] for img_id in imageid_list]))
    return torch.stack([prediction_cache[img_id] for img_id in imageid_list])


#-------------------------------------------------------------------------------------------------
def sigmoid(x):
    return 0 if (1 / (1 + math.exp(-x))) < 0.5 else 1

sigmoid_v = np.vectorize(sigmoid)

#-------------------------------------------------------------------------------------------------
# COMBINED 
def fitness_func(position):
    images = [unlabeled_pool[i] for i in range(len(unlabeled_pool)) if position[i]==1]
    p = get_prediction_probabilites(images)

    # LEAST CONFIDENCE
    if FITNESS_TYPE == "least_confidence":
        max_probs, max_indices = torch.max(p, axis=1)
        least_confidence = 1 - max_probs
        return torch.mean(least_confidence)  # avg least_confidence

    # ENTROPY
    elif FITNESS_TYPE == "entropy":
        entropies = torch.sum(- p * np.log2(p + 1e-07), axis=1)
        return torch.mean(entropies)   # avg total entropy

    # LOWEST MARGIN
    elif FITNESS_TYPE == "lowest_margin":
        p, _ = torch.sort(p, axis=1, descending=True)
        lowest_margin = p[:, 0] - p[:, 1]   # diff of max and 2nd max
        return torch.mean(lowest_margin)  # avg lowest_margin

    # HIGHEST MARGIN
    elif FITNESS_TYPE == "highest_margin":
        p, _ = torch.sort(p, axis=1, descending=True)
        highest_margin = p[:, 0] - p[:, -1]   # diff of max and min
        return torch.mean(highest_margin)  # avg highest_margin

#-------------------------------------------------------------------------------------------------
class Particle:
    def __init__(self, dimension, selection_probability=0.5):
        self.dimension = dimension

        self.position = np.array([1 if random.random()<selection_probability else 0 for _ in range(dimension)])     # particle position
        self.velocity = np.zeros(dimension)         # particle velocity
        self.pos_best = self.position               # best position individual

        self.fitness_best = -1          # best fitness individual
        self.fitness = -1               # fitness individual


    # evaluate current fitness
    def evaluate(self):
        self.fitness = fitness_func(self.position)

        # check to see if the current position is an individual best
        if self.fitness > self.fitness_best or self.fitness_best==-1:
            self.pos_best = self.position
            self.fitness_best = self.fitness


    # update new particle velocity
    def update_velocity(self, w, c1, c2, pos_best_g):

        for i in range(self.dimension):
            r1=random.random()
            r2=random.random()

            vel_cognitive = c1*r1*(self.pos_best[i]-self.position[i])
            vel_social = c2*r2*(pos_best_g[i]-self.position[i])
            self.velocity[i] = w*self.velocity[i] + vel_cognitive + vel_social


    # update the particle position based off new velocity updates
    def update_position(self):
        self.position = self.position + self.velocity
        self.position = sigmoid_v(self.position)

    def __repr__(self):
        return f"p:{self.position}, v:{self.velocity}"


#-------------------------------------------------------------------------------------------------
D = len(unlabeled_pool)
population_size = 50
max_generations = 50
max_stall_generations = 15
stall_count = 0

image_selection_probability = IMAGE_SELECTION_PROB

# CHANGED w - 22-04-2024
w_start = 0.4
w_end = 0.9

# CHANGED c1 c2 - 22-04-2024
c1_start = 0
c1_end = 4
c2_start = 4
c2_end = 0

print("Dimension =", D)


#-------------------------------------------------------------------------------------------------
def get_w(t):
    t = min(t, max_generations)
    return w_start - ((w_start-w_end)*t/max_generations)

def get_c1(t):
    t = min(t, max_generations)
    return c1_start - ((c1_start-c1_end)*t/max_generations)

def get_c2(t):
    t = min(t, max_generations)
    return c2_start - ((c2_start-c2_end)*t/max_generations)

logfilepath = os.path.join(output_dir, "al_logs.csv")

if not os.path.isfile(logfilepath):
    with open(logfilepath, "a") as logfile:
        logfile.write("generation,curr_gen_best_fitness,global_best_fitness\n")



#-------------------------------------------------------------------------------------------------
swarm = [Particle(dimension=D, selection_probability=image_selection_probability) for _ in range(population_size)]

curr_best_index = -1
curr_best_pos = []

global_best_fitness = -1
prev_global_best_fitness = -1
global_best_pos = []


for g in range(max_generations):

    print(f"\n=====================[ GENERATION {g+1} ]=====================")

    # evaluate eah group of images
    for i in range(population_size):
        swarm[i].evaluate()

    curr_gen_best_fitness = swarm[0].fitness
    for i in range(population_size):
        # determine if current particle is the best
        if swarm[i].fitness > curr_gen_best_fitness:
            curr_best_index = i
            curr_best_pos = np.copy(swarm[i].position)
            curr_gen_best_fitness = float(swarm[i].fitness)

            # determine if current particle is the best (globally)
            if curr_gen_best_fitness > global_best_fitness or global_best_fitness==-1:
                global_best_pos = np.copy(swarm[i].position)
                global_best_fitness = curr_gen_best_fitness

    # print all the particle fitness scores
    fitness_scores = [swarm[i].fitness for i in range(len(swarm))]
    print("fitness_scores:", fitness_scores)

    # cycle through swarm and update velocities and position
    w = get_w(g)
    c1 = get_c1(g)
    c2 = get_c2(g)
    print("w =", w, ", c1 =", c1, ", c2 =", c2)
    for i in range(population_size):
        swarm[i].update_velocity(w, c1, c2, global_best_pos)
        swarm[i].update_position()

    # save metrics in log file
    with open(logfilepath, "a") as logfile:
        logfile.write(f"{g+1},{curr_gen_best_fitness},{global_best_fitness}\n")

    print("best_fitness:    curr:", curr_gen_best_fitness, "    global:", global_best_fitness)

    # stop if fitness doesnt improve for a certain number of generations
    if max_stall_generations>=0:
        if global_best_fitness > prev_global_best_fitness:
            prev_global_best_fitness = global_best_fitness
            stall_count = 0
        else:
            stall_count += 1
            print("Fitness did not improve for", stall_count, "generations.")
            if stall_count >= max_stall_generations:
                print("Early stoppping PSO")
                break

selected_images = [unlabeled_pool[i] for i in range(D) if global_best_pos[i] == 1]

# add selected images to labeled pool
before_size = len(labeled_dataset)
print("len of labeled_dataset before:", len(labeled_dataset))
print("len of selected_images :", len(selected_images))
labeled_dataset.add_data(selected_images)
print("len of labeled_dataset after:", len(labeled_dataset), "( +", len(labeled_dataset)-before_size, ")")



#-------------------------------------------------------------------------------------------------
# unfreeze model weights
for param in model.parameters():
    param.requires_grad = True


# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-9,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
# -----------------------------------------------------------------------------------------------------------
# train the model
train_loss, train_acc, val_loss, val_acc = train_model(
    num_epochs=200,
    patience=15,            # CHANGED patience from 15 to 20 - 23-04-2024
    save_weights=True,
    output_dir=output_dir
)


# =====================================================================================================
"""# **Evaluate**"""

test_dataset = torchvision.datasets.ImageFolder(
    test_path,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMAGE_SIZE, antialias=True),
        torchvision.transforms.ToTensor()
    ])
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print("test loader size:", len(test_loader))

def evaluate(test_loader_obj):
    total_test_loss = 0
    total_test_acc = 0
    n_total_steps = len(test_loader_obj)

    model.eval()
    with torch.no_grad():
        steps = 0
        for x, y in test_loader_obj:
            steps+=1

            x = x.to(device)
            x = pretrained_preprocess(x)
            y = y.to(device)

            outputs = model(x)
            del x
            # Compute the loss and accuracy
            test_loss = loss_fn(outputs, y)
            test_acc = calc_acc(y, outputs)

            test_loss_number = test_loss.detach().item()
            test_acc_number = test_acc.detach().item()

            total_test_loss += test_loss_number
            total_test_acc  += test_acc_number

            if steps%100==0 or steps==1 or steps==n_total_steps:
                print(f"[{steps:4d}/{n_total_steps:4d}] loss: {(total_test_loss/steps):.4f}  acc: {(total_test_acc/steps):.4f}")

        total_test_loss   /= n_total_steps
        total_test_acc    /= n_total_steps

        # print(f"loss: {total_test_loss:.4f}   acc: {total_test_acc:.4f}")
        return  total_test_loss, total_test_acc

test_loss, test_acc = evaluate(test_loader)

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

#------------------------------------------------------------------------
# **Best Model Evaluate**

# load checkpoint if exists
best_epoch = -1     # 0 indexed, -1 is no epochs previously
best_checkpoint_filename = "model_best_checkpoint.pth"
save_file = os.path.join(output_dir, best_checkpoint_filename)
if os.path.isfile(save_file):
    checkpoint = torch.load(save_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_epoch = checkpoint['epoch']
    print("loaded best epoch",  best_epoch, "from", save_file)

best_test_loss, best_test_acc = evaluate(test_loader)

print("Best test loss:", best_test_loss)
print("Best test accuracy:", best_test_acc)


# save metrics in log file
evalfilepath = os.path.join(output_dir,"eval.csv")
with open(evalfilepath, "w") as logfile:
    logfile.write("test_loss,test_acc,best_test_loss,best_test_acc\n")
    logfile.write(f"{test_loss},{test_acc},{best_test_loss},{best_test_acc}\n")