import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import copy

np.random.seed(0)
torch.manual_seed(0)

rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 1, 30))
y = np.sin(2 * np.pi * X) + rng.normal(0, .3, 30)
Xv = np.sort(rng.uniform(0, 1, 30))
yv = np.sin(2 * np.pi * Xv) + rng.normal(0, .3, 30)

print(f"{'d':<5} | {'train MSE':<10} | {'val MSE':<10}")
for d in[1,4,15]:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(X[:, None], y)
    print(f"{d:<5} | {np.mean((m.predict(X[:, None]) - y)**2):<10.4f} | {np.mean((m.predict(Xv[:, None]) - yv)**2):<10.4f}")

def get_moons_data(n_samples=100):
    X_m, y_m = make_moons(n_samples=n_samples, noise=0.3, random_state=0)
    X_tr, X_va, y_tr, y_va = train_test_split(X_m, y_m, test_size=0.5, random_state=0)
    return (torch.FloatTensor(X_tr), torch.FloatTensor(X_va), 
            torch.FloatTensor(y_tr), torch.FloatTensor(y_va))

Xtr, Xva, ytr, yva = get_moons_data(100)
loss_fn = nn.BCEWithLogitsLoss()

def get_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X).squeeze(1)
        preds = (torch.sigmoid(logits) > 0.5).int()
        return accuracy_score(y.numpy(), preds.numpy())

def fit(model, opt, steps=2000):
    tr, va = [], []
    for i in range(steps):
        model.train()
        opt.zero_grad()
        l = loss_fn(model(Xtr).squeeze(1), ytr)
        l.backward()
        opt.step()
        if i % 50 == 0:
            model.eval()
            with torch.no_grad():
                tr.append(l.item())
                va.append(loss_fn(model(Xva).squeeze(1), yva).item())
    return tr, va

big = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))
tr, va = fit(big, torch.optim.Adam(big.parameters(), 1e-2))
plt.plot(tr, label='train')
plt.plot(va, label='val')
plt.legend()
plt.show()

def fit_early_stopping(model, opt, steps=2000, patience=10):
    tr, va = [], []
    best_loss = float('inf')
    best_weights = None
    patience_counter = 0
    for i in range(steps):
        model.train()
        opt.zero_grad()
        l = loss_fn(model(Xtr).squeeze(1), ytr)
        l.backward()
        opt.step()
        if i % 50 == 0:
            model.eval()
            with torch.no_grad():
                tr_l = l.item()
                va_l = loss_fn(model(Xva).squeeze(1), yva).item()
                tr.append(tr_l)
                va.append(va_l)
                if va_l < best_loss:
                    best_loss = va_l
                    best_weights = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return tr, va

model_es = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))
tr_es, va_es = fit_early_stopping(model_es, torch.optim.Adam(model_es.parameters(), 1e-2))

model_wd = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))
tr_wd, va_wd = fit(model_wd, torch.optim.Adam(model_wd.parameters(), 1e-2, weight_decay=1e-2))

reg = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128,128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128,1))
tr2, va2 = fit(reg, torch.optim.Adam(reg.parameters(), 1e-2, weight_decay=1e-2))

Xtr_old, Xva_old, ytr_old, yva_old = Xtr, Xva, ytr, yva
Xtr, Xva, ytr, yva = get_moons_data(2000)
big_large = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))
tr_large, va_large = fit(big_large, torch.optim.Adam(big_large.parameters(), 1e-2))

plt.plot(tr_large, label='train large')
plt.plot(va_large, label='val large')
plt.legend()
plt.show()

print(f"{'Method':<20} | {'Val Loss':<10} | {'Val Acc':<10}")
print("-" * 48)
print(f"{'No reg':<20} | {va[-1]:<10.4f} | {get_accuracy(big, Xva_old, yva_old):<10.2%}")
print(f"{'Early stop':<20} | {va_es[-1]:<10.4f} | {get_accuracy(model_es, Xva_old, yva_old):<10.2%}")
print(f"{'Weight decay':<20} | {va_wd[-1]:<10.4f} | {get_accuracy(model_wd, Xva_old, yva_old):<10.2%}")
print(f"{'Dropout':<20} | {va2[-1]:<10.4f} | {get_accuracy(reg, Xva_old, yva_old):<10.2%}")
print(f"{'More data':<20} | {va_large[-1]:<10.4f} | {get_accuracy(big_large, Xva, yva):<10.2%}")
