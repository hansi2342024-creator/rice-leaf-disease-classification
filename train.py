#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rice Leaf Disease Classification using Deep Learning

This project implements an end-to-end image classification pipeline
for detecting rice leaf diseases using CNN and transfer learning (ResNet18, ResNet50).

Key features:
- Data preprocessing and cleaning
- Stratified train/validation/test split
- Data augmentation
- Model training and evaluation
- Multi-scenario testing (white / field / mixed)
- Performance metrics and confusion matrices
"""
import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# -------------------------- Config --------------------------
DATASET_ROOT = os.path.join(".", "Dhan-Shomadhan")
IMAGE_SIZE = 224
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LABEL_NAMES = ['BrownSpot', 'LeafScaled', 'RiceBlast', 'RiceTungro', 'SheathBlight']

# -------------------------- Utils ---------------------------
def is_image_file(path):
    return os.path.splitext(path)[1].lower() in {'.jpg', '.jpeg', '.png'}

def normalize_class_name(name):
    mapping = {
        "brownspot": "BrownSpot", "browonspot": "BrownSpot", "brown spot": "BrownSpot", "browon spot":"BrownSpot",
        "leafscaled": "LeafScaled", "leaf scaled": "LeafScaled",
        "riceblast": "RiceBlast", "rice blast": "RiceBlast",
        "ricetungro": "RiceTungro", "rice tungro": "RiceTungro", "rice turgro":"RiceTungro",
        "sheath blight":"SheathBlight","shath blight":"SheathBlight","sheathblight":"SheathBlight",
    }
    key = name.lower().strip()
    if key in mapping:
        return mapping[key]
    for k,v in mapping.items():
        if k in key:
            return v
    return ''.join(w.capitalize() for w in name.split())

def detect_background_from_path(path):
    p = path.replace("\\", "/").lower()
    if ("whitebackground" in p) or ("/white/" in p) or ("white background" in p):
        return "white"
    if ("fieldbackground" in p) or ("/field/" in p) or ("field background" in p):
        return "field"
    return "unknown"

def detect_label_from_path(path):
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    for part in reversed(parts):
        normalized = normalize_class_name(part)
        if normalized in LABEL_NAMES:
            return normalized
    parent = os.path.basename(os.path.dirname(path))
    if parent:
        return normalize_class_name(parent)
    return "unknown"

def scan_dataset(root_path):
    print(f"Scanning: {root_path}")
    all_images = []
    for root, _, files in os.walk(root_path):
        for fn in files:
            if fn.startswith("._"):
                continue
            fp = os.path.join(root, fn)
            if is_image_file(fp):
                all_images.append(fp)
    rows = []
    for p in all_images:
        rows.append({"path": p, "label": detect_label_from_path(p), "bg": detect_background_from_path(p)})
    df = pd.DataFrame(rows)
    df = df[df.label != "unknown"].reset_index(drop=True)
    print(f"Total images used: {len(df)}")
    return df

class PathsDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.paths[idx]

# ----------------------- Models -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*14*14, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def build_model(name, num_classes, pretrained=True):
    name = name.lower()
    if name == "cnn":
        return SimpleCNN(num_classes)
    if name == "resnet18":
        m = models.resnet18(pretrained=pretrained); m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if name == "resnet50":
        m = models.resnet50(pretrained=pretrained); m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    raise ValueError("Unknown model: " + name)

# ----------------------- Train/Eval -------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    losses, preds_all, labels_all = [], [], []
    for imgs, labels, _ in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        losses.append(loss.item())
        preds_all.extend(outputs.argmax(1).detach().cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())
    return float(np.mean(losses)), accuracy_score(labels_all, preds_all)

def evaluate(model, loader):
    model.eval()
    preds_all, labels_all, paths = [], [], []
    with torch.no_grad():
        for imgs, labels, ps in tqdm(loader, desc="Evaluating", leave=False):
            outputs = model(imgs.to(DEVICE))
            preds = outputs.argmax(1).detach().cpu().numpy()
            preds_all.extend(preds); labels_all.extend(labels.numpy()); paths.extend(ps)
    acc = accuracy_score(labels_all, preds_all)
    prec, rec, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average='macro', zero_division=0)
    cm = confusion_matrix(labels_all, preds_all)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm, "preds": preds_all, "labels": labels_all, "paths": paths}

def run_one_split(df, num_classes, seed, hyper, model_name):
    set_seed(seed)
    # Transforms
    class ResizeWithPadding:
        def __init__(self, target=224): self.target = target
        def __call__(self, img: Image.Image):
            w, h = img.size; s = self.target / max(w, h)
            nw, nh = int(w*s), int(h*s); img2 = img.resize((nw, nh), Image.BILINEAR)
            bg = Image.new("RGB", (self.target, self.target)); bg.paste(img2, ((self.target-nw)//2, (self.target-nh)//2))
            return bg
    if model_name == "cnn":
        train_tf = transforms.Compose([ResizeWithPadding(IMAGE_SIZE),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomRotation(15),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        val_tf   = transforms.Compose([ResizeWithPadding(IMAGE_SIZE),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    else:
        train_tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomRotation(15),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        val_tf   = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    # Split
    idxs = np.arange(len(df)); y = df["label_idx"].values
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(s1.split(idxs, y))
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    tr_rel, va_rel = next(s2.split(trainval_idx, y[trainval_idx]))
    train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[va_rel]
    print(f"[{model_name}] Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    # Data
    train_ds = PathsDataset(df.iloc[train_idx]["path"].tolist(), df.iloc[train_idx]["label_idx"].tolist(), transform=train_tf)
    val_ds   = PathsDataset(df.iloc[val_idx]["path"].tolist(),   df.iloc[val_idx]["label_idx"].tolist(),   transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=hyper["BATCH_SIZE"], shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=hyper["BATCH_SIZE"], shuffle=False, num_workers=0)

    # Model
    model = build_model(model_name, num_classes, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper["LR"])

    best_val_acc, best_state = -1.0, None
    for ep in range(hyper["NUM_EPOCHS"]):
        loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vm = evaluate(model, val_loader); va = vm["accuracy"]
        if va > best_val_acc:
            best_val_acc = va; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{model_name}] Epoch {ep+1}/{hyper['NUM_EPOCHS']} Loss {loss:.4f} | TrainAcc {tr_acc:.4f} | ValAcc {va:.4f}")
    if best_state: model.load_state_dict(best_state)

    # Test scenarios
    test_df = df.iloc[test_idx].reset_index(drop=True)
    scenarios = {}
    for name, cond in [("white", lambda r: r["bg"] == "white"),
                       ("field", lambda r: r["bg"] == "field"),
                       ("mixed", lambda r: True)]:
        sel = test_df[test_df.apply(cond, axis=1)].reset_index(drop=True)
        if len(sel) == 0: scenarios[name] = None; continue
        ds = PathsDataset(sel["path"].tolist(), sel["label_idx"].tolist(), transform=val_tf)
        loader = DataLoader(ds, batch_size=hyper["BATCH_SIZE"], shuffle=False, num_workers=0)
        scenarios[name] = evaluate(model, loader)
    return scenarios

def run_repeats(df, num_classes, model_name, k, hyper):
    seeds = [SEED + i*13 for i in range(k)]
    all_runs = []
    for i, seed in enumerate(seeds):
        print("="*50); print(f"[{model_name}] Repeat {i+1}/{k} seed={seed}")
        all_runs.append(run_one_split(df, num_classes, seed, hyper, model_name))

    def collect(metric, scen):
        vals = []
        for r in all_runs:
            sc = r.get(scen)
            if sc is None: continue
            vals.append(sc.get(metric))
        if len(vals)==0: return None
        arr = np.array(vals)
        return float(arr.mean()), float(arr.std())

    summary = {sc: {m: collect(m, sc) for m in ["accuracy","precision","recall","f1"]}
               for sc in ["white","field","mixed"]}
    return summary, all_runs

# ----------------------- Plots ------------------------------
def save_confusion_matrices(last_run, model_name):
    import matplotlib.pyplot as plt, numpy as np
    os.makedirs("figs", exist_ok=True)
    for scen in ["white","field","mixed"]:
        s = last_run.get(scen)
        if not s or s.get("cm") is None: continue
        cm = s["cm"]
        plt.figure(figsize=(5.5, 4.8))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"{model_name} Confusion Matrix - {scen.title()}")
        plt.colorbar()
        ticks = np.arange(len(LABEL_NAMES))
        plt.xticks(ticks, LABEL_NAMES, rotation=45, ha="right")
        plt.yticks(ticks, LABEL_NAMES)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout()
        out = f"figs/{model_name.lower()}_cm_{scen}.png"
        plt.savefig(out, dpi=300); plt.close()
        print(f"Saved: {out}")

def plot_bar_comparison(results_by_model, scenario="mixed"):
    import matplotlib.pyplot as plt
    metrics = ["accuracy","precision","recall","f1"]
    models = list(results_by_model.keys())
    means = [[results_by_model[m][scenario][k][0] if results_by_model[m][scenario][k] else 0.0 for m in models] for k in metrics]
    x = np.arange(len(models))
    os.makedirs("figs", exist_ok=True)
    for i, met in enumerate(metrics):
        plt.figure(figsize=(6.2, 4.0))
        plt.bar(x, means[i]); plt.xticks(x, models)
        plt.ylim(0, 1.0); plt.ylabel(met.title()); plt.title(f"{scenario.title()} - {met.title()}")
        out = f"figs/bar_{scenario}_{met}.png"; plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
        print(f"Saved: {out}")

# ----------------------- Main -------------------------------
# Entry point for training and evaluation pipeline
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["cnn","resnet18","resnet50","all"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print("Rice Leaf Disease Classification - Training Pipeline")
    df = scan_dataset(DATASET_ROOT)
    labels_unique = sorted(df.label.unique())
    label2idx = {lab:i for i,lab in enumerate(labels_unique)}
    df["label_idx"] = df.label.map(label2idx)
    num_classes = len(labels_unique)
    print("Classes:", label2idx)

    hypers = {
        "cnn":      {"BATCH_SIZE": 32, "NUM_EPOCHS": 30, "LR": 5e-4},
        "resnet18": {"BATCH_SIZE": 32, "NUM_EPOCHS": 2,  "LR": 1e-4},
        "resnet50": {"BATCH_SIZE": 16, "NUM_EPOCHS": 2,  "LR": 1e-4},
    }
    to_run = ["cnn","resnet18","resnet50"] if args.model=="all" else [args.model]

    results_by_model = {}
    raw_runs = {}
    for m in to_run:
        print("#"*40); print("Model:", m)
        summary, raw = run_repeats(df, num_classes, m, k=args.repeats, hyper=hypers[m])
        results_by_model[m] = summary; raw_runs[m] = raw
        save_confusion_matrices(raw[-1], model_name=m.title())
        print("-- Summary (mean±std) over", args.repeats, "runs --")
        for scen, mets in summary.items():
            if mets["accuracy"] is None:
                print(scen, ": no samples"); continue
            print("Scenario:", scen)
            for k,v in mets.items():
                mu, sd = v if v else (0.0, 0.0)
                print(f"  {k:9s}: {mu:.4f} ± {sd:.4f}")

    if len(results_by_model) > 1:
        plot_bar_comparison(results_by_model, scenario="mixed")
        print("Saved bar charts to ./figs/")
    print("Done.")

if __name__ == "__main__":
    main()
