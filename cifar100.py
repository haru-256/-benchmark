import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(prog="benchmark to use pytorch lib", add_help=True)
parser.add_argument(
    "-g",
    "--gpu_id",
    help="specify gpu by this number. defalut value is 0," " -1 is means don't use gpu",
    choices=[-1, 0, 1],
    type=int,
    default=0,
)
parser.add_argument("-e", "--epoch", help="number of epochs")
parser.add_argument("-bs", "--batch_size", help="number of batch size")
opt = parser.parse_args()
# device
if opt.gpu_id == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:{}".format(opt.gpu_id))
# データの定義
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}
trainset = datasets.CIFAR100(
    root="./data", train=True, download=True, transform=data_transforms["train"]
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=opt.batch_size, shuffle=True, num_workers=6
)
valset = datasets.CIFAR100(
    root="./data", train=False, download=True, transform=data_transforms["train"]
)
testloader = torch.utils.data.DataLoader(
    valset, batch_size=opt.batch_size, shuffle=False, num_workers=6
)
dataloaders = {"train": trainloader, "val": testloader}
dataset_sizes = {phase: len(dataloaders[phase]) for phase in ["train", "val"]}

# モデル
model = models.resnet101(pretrained=True, progress=True)
num_in = model.fc.in_features
model.fc = nn.Linear(num_in, 100)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, momentum=0.9)

# loss
criterion = nn.CrossEntropyLoss()
since = time.time()
for epoch in tqdm(range(opt.epoch), desc="Epoch", unit="epoch"):
    # Each epoch has a training and validation phase
    for phase in tqdm(["train", "val"], desc="Phase"):

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in tqdm(dataloaders[phase], desc="Iteration", unit="iter"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == "train"):
                logits = model(inputs)
                loss = criterion(logits, labels)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        tqdm.write("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

time_elapsed = time.time() - since
print(
    "Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
)
