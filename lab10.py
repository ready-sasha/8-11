import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.set_num_threads(4)

def main():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('data', train=True, download=True, transform=tf)
    test_data = datasets.MNIST('data', train=False, download=True, transform=tf)
    
    train_dl = DataLoader(train_data, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=512)
    
    fig, ax = plt.subplots(5, 5, figsize=(6, 6))
    for i, a in enumerate(ax.flat):
        a.imshow(train_data[i][0][0], cmap='gray')
        a.set_title(train_data[i][1])
        a.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_grid.png')
    plt.close()

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

    for ep in range(3):
        t0 = time.time()
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_dl:
                correct += (model(x).argmax(1) == y).sum().item()
        
        print(f'epoch {ep} loss {loss.item():.3f} acc {correct / len(test_data):.4f} lr {sched.get_last_lr()[0]} time {time.time()-t0:.0f}s')
        sched.step()

    model.eval()
    errors = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_dl:
            preds = model(x).argmax(1)
            all_preds.extend(preds.numpy())
            all_targets.extend(y.numpy())
            
            misclassified = (preds != y).nonzero(as_tuple=True)[0]
            for idx in misclassified:
                if len(errors) < 16:
                    errors.append({
                        'img': x[idx][0].numpy(),
                        'pred': preds[idx].item(),
                        'true': y[idx].item()
                    })

    fig, ax = plt.subplots(4, 4, figsize=(8, 8))
    for i, a in enumerate(ax.flat):
        if i < len(errors):
            a.imshow(errors[i]['img'], cmap='gray')
            a.set_title(f"{errors[i]['pred']} / {errors[i]['true']}")
        a.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_errors.png')
    plt.close()

    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1
        
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', 
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.tight_layout()
    plt.savefig('mnist_cm.png')
    plt.close()

    torch.save(model.state_dict(), 'mnist.pt')
    
    m2 = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    m2.load_state_dict(torch.load('mnist.pt'))
    m2.eval()
    
    correct2 = 0
    with torch.no_grad():
        for x, y in test_dl:
            correct2 += (m2(x).argmax(1) == y).sum().item()
    print(f'm2 verification accuracy: {correct2 / len(test_data):.4f}')

if __name__ == '__main__':
    main()
