import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from src.cv_dataset import XBDDamageDataset

def get_device():
    # Use Apple GPU (MPS) if available, else CPU
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train(data_dir="disaster-ai/data/xbd", outdir="disaster-ai/models/cv", img=224, bs=8, lr=2e-4, epochs=2):
    device = get_device()
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds_tr = XBDDamageDataset(data_dir, split="train", img_size=img)
    ds_va = XBDDamageDataset(data_dir, split="val", img_size=img)

    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False)

    # Define model
    model = resnet18(weights=None)  
    model.fc = nn.Linear(model.fc.in_features, 4)  # 4 damage classes
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(epochs):
        # Training loop
        model.train(); tot=0
        for x,y in dl_tr:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item()*x.size(0)
        tr_loss = tot/len(dl_tr.dataset)

        # Validation
        model.eval(); correct=0
        with torch.no_grad():
            for x,y in dl_va:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item()
        acc = correct/len(dl_va.dataset)
        print(f"epoch {ep+1}: train_loss {tr_loss:.4f}  val_acc {acc:.3f}")

        # Save best model
        if acc > best:
            best = acc
            torch.save(model.state_dict(), outdir/"resnet18_xbd.pt")
    print("Best validation accuracy:", best)

if __name__ == "__main__":
    train()
