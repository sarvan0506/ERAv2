import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms

def get_device(device_option='cuda'):
    
    if device_option == 'cuda':
        use_cuda = torch.cuda.is_available()
        return torch.device("cuda" if use_cuda else "cpu")
    else:
        return torch.device("cpu")
    
def diplay_batch(data_loader, cmap='gray'):

    batch_data, batch_label = next(iter(data_loader))

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap=cmap)
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])
        
def load_dataset(train=True):
    
    if train:

        train_transforms = transforms.Compose([
            transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation((-15., 15.), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        return datasets.MNIST('../data', train=True, download=True, transform=train_transforms)

    else:

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        return datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion, train_losses, train_acc):
    
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()train_losses, train_acc, test_losses, test_acc
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
    
def test(model, device, test_loader, criterion, test_losses, test_acc):
    
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def train_model(model, device, train_loader, test_loader, criterion, optimizer, scheduler, epoch):
    
   
    # Data to plot accuracy and loss graphs
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    

    test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}
    
    for epoch in range(1, epoch+1):
        print(f'Epoch {epoch}')
        train(model, device, train_loader, optimizer, criterion, train_losses, train_acc)
        test(model, device, test_loader, criterion, test_losses, test_acc)
        scheduler.step()
        
    return train_losses, train_acc, test_losses, test_acc

def plot_model_metrics(train_losses, train_acc, test_losses, test_acc):
    
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")