"""# **Imports**"""

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
# parser.add_argument('-f', '--fitness_type', choices=['least_confidence', 'entropy', 'lowest_margin', 'highest_margin'])
parser.add_argument('-m', '--model_name', choices=['efficientnetv2m', 'efficientnetv2s', 'densenet121', 'mobilenetv2', 'resnet101', 'vgg16'])
# parser.add_argument('-w', '--weight_decay', type=float)
# parser.add_argument('-s', '--selection_probability', type=float)
parser.add_argument('-b', '--batch_size', type=int, default=8)


args = parser.parse_args()
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
BATCH_SIZE = args.batch_size
MODEL_NAME = args.model_name

output_dir = f"./save_files_simple_tl/{MODEL_NAME}_simple_bs{BATCH_SIZE}"
os.makedirs(output_dir, exist_ok = True)        # create directory if it doesnt exist

output_dir_transfer_learning = os.path.join(output_dir, "transfer_learning")
output_dir_fine_tuning = os.path.join(output_dir, "fine_tuning")
os.makedirs(output_dir_transfer_learning, exist_ok = True) 
os.makedirs(output_dir_fine_tuning, exist_ok = True) 


class_names = sorted(os.listdir(train_path))
num_classes = len(class_names)

print("-------------------------------", output_dir.split("/")[-1], "-------------------------------")

print(num_classes, "classes:", class_names)


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

# create a dictionary to create a numerical representation of each label
label2idx = {class_names[i]:i for i in range(num_classes)}
print(label2idx)



train_dataset = torchvision.datasets.ImageFolder(
    train_path,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMAGE_SIZE, antialias=True),
        torchvision.transforms.ToTensor()
    ])
)

# generator1 = torch.Generator().manual_seed(42)
# train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1], generator=generator1)

val_dataset = torchvision.datasets.ImageFolder(
    val_path,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMAGE_SIZE, antialias=True),
        torchvision.transforms.ToTensor()
    ])
)

print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print("Train loader size:", len(train_loader))

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print("val loader size:", len(val_loader))


#-------------------------------------------------------------------------------------------------
"""# **CNN model**"""

weights = None
model = None
pretrained_preprocess = None


if MODEL_NAME == "efficientnetv2m":

    weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT          # pretrained weights for model
    model = torchvision.models.efficientnet_v2_m(weights=weights)           # define the model with pretrained weights

    print(model)

    # get the transformations required for
    # preprocessing images before sending to pretrained model
    pretrained_preprocess = weights.transforms()

    # freeze model pretrained weights
    for param in model.parameters():
        param.requires_grad = False

    # replace last layer
    num_ftrs = model.classifier[1].in_features
    print("num_ftrs: ", num_ftrs)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 128),
        nn.ReLU(True),
        nn.Dropout(p=0.5), 
        nn.Linear(128, num_classes),
        nn.Softmax(dim = -1)
    )

elif MODEL_NAME == "densenet121":
    weights = torchvision.models.DenseNet121_Weights.DEFAULT          # pretrained weights for model
    model = torchvision.models.densenet121(weights=weights)           # define the model with pretrained weights

    print(model)

    # get the transformations required for
    # preprocessing images before sending to pretrained model
    pretrained_preprocess = weights.transforms()

    # freeze model pretrained weights
    for param in model.parameters():
        param.requires_grad = False

    # replace last layer
    num_ftrs = model.classifier.in_features

    print("num_ftrs: ", num_ftrs)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 128),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(128, num_classes),
        nn.Softmax(dim = -1)
    )

elif MODEL_NAME == "efficientnetv2s":
    weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT          # pretrained weights for model
    model = torchvision.models.efficientnet_v2_s(weights=weights)           # define the model with pretrained weights

    print(model)

    # get the transformations required for
    # preprocessing images before sending to pretrained model
    pretrained_preprocess = weights.transforms()

    # freeze model pretrained weights
    for param in model.parameters():
        param.requires_grad = False

    # replace last layer
    num_ftrs = model.classifier[1].in_features
    print("num_ftrs: ", num_ftrs)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 128),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(128, num_classes),
        nn.Softmax(dim = -1)
    )


elif MODEL_NAME == "mobilenetv2":

    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT          # pretrained weights for model
    model = torchvision.models.mobilenet_v2(weights=weights)           # define the model with pretrained weights

    print(model)

    # get the transformations required for
    # preprocessing images before sending to pretrained model
    pretrained_preprocess = weights.transforms()

    # freeze model pretrained weights
    for param in model.parameters():
        param.requires_grad = False

    # replace last layer

    num_ftrs = model.classifier[1].in_features
    print("num_ftrs: ", num_ftrs)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 128),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(128, num_classes),
        nn.Softmax(dim = -1)
    )


elif MODEL_NAME == "resnet101":
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


elif MODEL_NAME == "vgg16":
    weights = torchvision.models.VGG16_Weights.DEFAULT          # pretrained weights for model
    model = torchvision.models.vgg16(weights=weights)           # define the model with pretrained weights

    print(model)

    # get the transformations required for
    # preprocessing images before sending to pretrained model
    pretrained_preprocess = weights.transforms()

    # freeze model pretrained weights
    for param in model.parameters():
        param.requires_grad = False

    # # replace last layer
    num_ftrs = model.classifier[0].in_features
    print("num_ftrs: ", num_ftrs)

    # model.fc = nn.Sequential(
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(128, num_classes),
        nn.Softmax(dim = -1)
    )

print("model created...")
model = model.to(device)
print("model loaded...")

# print(model)


loss_fn = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-9,
    weight_decay=1e-4
)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

#-------------------------------------------------------------------------------------------------
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

        # scheduler.step(val_loss)    # adjust learning rate

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
# TRANSFER LEARNING

train_model(
    num_epochs=100, #CHANGED from 50 to 100
    patience=10,    #CHANGED from 5 to 10
    save_weights=True,
    output_dir=output_dir_transfer_learning
)

# #-------------------------------------------------------------------------------------------------
# # FINE TUNING

# # unfreeze model weights
# for param in model.parameters():
#     param.requires_grad = True


# # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=0.5 * 1e-4,
#     betas=(0.9, 0.999),
#     eps=1e-9,
#     # weight_decay=1e-4
# )

# # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

# # train the model
# train_loss, train_acc, val_loss, val_acc = train_model(
#     num_epochs=200,
#     patience=10,
#     save_weights=True,
#     output_dir=output_dir_fine_tuning
# )

#-------------------------------------------------------------------------------------------------
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
save_file = os.path.join(output_dir_fine_tuning, best_checkpoint_filename)
if os.path.isfile(save_file):
    checkpoint = torch.load(save_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_epoch = checkpoint['epoch']
    print("loaded best epoch",  best_epoch, "from", save_file)

best_test_loss, best_test_acc = evaluate(test_loader)

print("Best test loss:", best_test_loss)
print("Best test accuracy:", best_test_acc)

#------------------------------------------------------------------------
# save metrics in log file
evalfilepath = os.path.join(output_dir,"eval.csv")
with open(evalfilepath, "w") as logfile:
    logfile.write("test_loss,test_acc,best_test_loss,best_test_acc\n")
    logfile.write(f"{test_loss},{test_acc},{best_test_loss},{best_test_acc}\n")