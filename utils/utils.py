import torch
from torchvision import datasets, transforms
from utils.augmentations import RandAugment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def get_loaders(data_directory, batch_size, augment=True, N=2, M=9): # only support imagenet-size image
    print('==> Preparing dataset..')
    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std), 
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    
    if augment:
        # Add RandAugment with N, M(hyperparameter)
        train_transform.transforms.insert(0, RandAugment(N, M))

    train_dataset = datasets.ImageFolder(root=data_directory+'/train', \
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=12, pin_memory=True)
    test_dataset = datasets.ImageFolder(root=data_directory+'/val', \
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=12, pin_memory=True)
    return train_loader, test_loader
    
# note, why normalize image separately? because in attack phase we don't want the normalization mess with the attack budget
# (epsilon / std) but someone just don't care, besides we don't use that measurement.
# remove this has no impact on training time
# def normalize(image, batch_size):
#     mean = torch.tensor(imagenet_mean).reshape(1, 3, 1, 1).repeat(batch_size, 1, 1, 1).to(device)
#     std = torch.tensor(imagenet_std).reshape(1, 3, 1, 1).repeat(batch_size, 1, 1, 1).to(device)
#     return (image-mean)/ std