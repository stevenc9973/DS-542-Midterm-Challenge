import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import wandb
import json
from torch.utils.data import random_split

################################################################################
# Model Definition - Using Pretrained ResNet18
################################################################################
class FineTunedResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(FineTunedResNet, self).__init__()
        # Use Resnet50
        self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        for param in self.resnet.parameters():
            param.requires_grad = False

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),  
            nn.BatchNorm1d(1024),           
            nn.ReLU(),
            nn.Dropout(0.3),              
            nn.Linear(1024, num_classes))   
    
    def forward(self, x):
        return self.resnet(x)

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        # Output and backward step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Predict based on output data
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})
        
    # Calculate train_loss
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def main():
    ############################################################################
    #    Configuration Dictionary
    ############################################################################
    CONFIG = {
        "model": "FineTunedResNet",
        "batch_size": 256,
        "learning_rate": 0.0005,
        "epochs": 15,
        "num_workers": 4,
        "weight_decay": 0.0001, 
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    # Set random seed for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    ############################################################################
    #      Data Transformation
    ############################################################################

    # Radom crop to avoid overfitting
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]) # data from mean and deviation for CIFAR-100

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    ############################################################################
    #       Data Loading
    ############################################################################
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)

    # Split train into train and validation (80/20 split)
    train_size = int(0.8 * len(trainset))
    val_size = int(0.2 * len(trainset))
    trainset, valset = random_split(trainset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"],
                                             shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"],
                                           shuffle=False, num_workers=0)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"],
                                            shuffle=False, num_workers=0)
    
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    model = FineTunedResNet(num_classes=100)
    model = model.to(CONFIG["device"])

    print("\nModel summary:")
    print(f"{model}\n")

    ############################################################################
    # Loss Function, Optimizer and learning rate scheduler
    ############################################################################
    # Criterion based on entropy loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                       weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Initialize wandb
    #wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    #wandb.watch(model)

    ############################################################################
    # Training Loop
    ############################################################################
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        # Gradually unfreeze layers
        if epoch == 3:
            for param in model.resnet.layer4.parameters():
                param.requires_grad = True
        if epoch == 7:
            for param in model.resnet.layer3.parameters():
                param.requires_grad = True
        if epoch == 12:
            for param in model.resnet.parameters():
                param.requires_grad = True

        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # log to WandB
        '''
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })
        '''

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            #wandb.save("best_model.pth")

    #wandb.finish()

    ############################################################################
    # Evaluation
    ############################################################################
    import eval_cifar100
    import eval_ood

    # Load best model for evaluation
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()