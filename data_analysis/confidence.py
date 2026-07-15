import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

bert_model = "roberta-base"
def calibration_metrics(probs, labels, n_bins=20):
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    mce = 0.0
    rmsce = 0.0
    
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        
        avg_conf = probs[mask].mean()
        avg_acc  = labels[mask].mean()
        error = abs(avg_conf - avg_acc)

        ece += (mask.sum() / len(probs)) * error
        mce = max(mce, error)
        rmsce += (mask.sum() / len(probs)) * (error * error)

    rmsce = np.sqrt(rmsce)
    return ece, mce, rmsce

def eval_baseline(model, loader):
    model.eval()
    probs, trues = [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].cuda()
            mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cpu().numpy()

            p = model(ids, mask).cpu().numpy().flatten()

            probs.extend(p)
            trues.extend(labels)

    probs = np.array(probs)
    trues = np.array(trues)
    preds = (probs > 0.5).astype(int)

    f1 = f1_score(trues, preds)
    ece, mce, rmsce = calibration_metrics(probs, trues)

    return {"f1": f1, "ece": ece, "mce": mce, "rmsce": rmsce, "probs": probs}

class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

def learn_temperature(model, loader):
    model.eval()
    logits_list, labels_list = [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].cuda()
            mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            outputs = model.roberta(ids, mask)
            logits = model.fc(outputs.last_hidden_state[:,0,:])

            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    scaler = TemperatureScaler().cuda()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits)
        probs = torch.sigmoid(scaled).squeeze()
        loss = F.binary_cross_entropy(probs, labels.float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler

def eval_temperature_scaled(model, loader, scaler):
    model.eval()
    probs, trues = [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].cuda()
            mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cpu().numpy()

            outputs = model.roberta(ids, mask)
            logits = model.fc(outputs.last_hidden_state[:,0,:])
            scaled_logits = scaler(logits)
            p = torch.sigmoid(scaled_logits).cpu().numpy().flatten()

            probs.extend(p)
            trues.extend(labels)

    probs = np.array(probs)
    trues = np.array(trues)
    preds = (probs > 0.5).astype(int)

    f1 = f1_score(trues, preds)
    ece, mce, rmsce = calibration_metrics(probs, trues)

    return {"f1": f1, "ece": ece, "mce": mce, "rmsce": rmsce, "probs": probs}

def enable_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

def eval_mc_dropout(model, loader, passes=10):
    model.eval()
    model.apply(enable_dropout)  # turn on dropout at inference

    probs_all, trues = [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].cuda()
            mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cpu().numpy()

            mc_probs = []
            for _ in range(passes):
                p = model(ids, mask).cpu().numpy().flatten()
                mc_probs.append(p)

            mean_probs = np.mean(mc_probs, axis=0)
            probs_all.extend(mean_probs)
            trues.extend(labels)

    probs_all = np.array(probs_all)
    trues = np.array(trues)
    preds = (probs_all > 0.5).astype(int)

    f1 = f1_score(trues, preds)
    ece, mce, rmsce = calibration_metrics(probs_all, trues)

    return {"f1": f1, "ece": ece, "mce": mce, "rmsce": rmsce, "probs": probs_all}

def eval_ensemble(models, loader):
    all_probs = []
    trues = []

    for batch in loader:
        ids = batch["input_ids"].cuda()
        mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cpu().numpy()

        model_probs = []
        with torch.no_grad():
            for model in models:
                p = model(ids, mask).cpu().numpy().flatten()
                model_probs.append(p)

        mean_probs = np.mean(np.array(model_probs), axis=0)
        all_probs.extend(mean_probs)
        trues.extend(labels)

    all_probs = np.array(all_probs)
    trues = np.array(trues)
    preds = (all_probs > 0.5).astype(int)

    f1 = f1_score(trues, preds)
    ece, mce, rmsce = calibration_metrics(all_probs, trues)

    return {"f1": f1, "ece": ece, "mce": mce, "rmsce": rmsce, "probs": all_probs}

def evaluate_all(model, valid_loader, test_loader, ensemble_models=None):
    print("\n=== BASELINE ===")
    base = eval_baseline(model, test_loader)
    print(base)

    print("\n=== TEMPERATURE SCALING ===")
    scaler = learn_temperature(model, valid_loader)
    temp = eval_temperature_scaled(model, test_loader, scaler)
    print(temp)

    print("\n=== MONTE CARLO DROPOUT ===")
    mc = eval_mc_dropout(model, test_loader)
    print(mc)

    if ensemble_models is not None:
        print("\n=== ENSEMBLES ===")
        ens = eval_ensemble(ensemble_models, test_loader)
        print(ens)


def safe_text(x):
        if pd.isna(x) or x is None:
            return ""
        if isinstance(x, (float, int)):
            return str(x)
        if isinstance(x, (list, tuple)):
            return " ".join(map(str, x))
        return str(x)

class Dataset(DataLoader):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
        self.tok = tokenizer
        self.max_len = max_len
    
    def serialize(self, sample):
        string = ''
        name = safe_text(sample.get("name", ""))
        brand = safe_text(sample.get("brand", ""))
        desc = safe_text(sample.get("desc", ""))
        price = safe_text(sample.get("price", ""))

        string = f"{string}[COL] brand [VAL] {' '.join(brand.split(' ')[:5])}".strip()
        string = f"{string} [COL] name [VAL] {' '.join(name.split(' ')[:50])}".strip()
        string = f"{string} [COL] price [VAL] {price}".strip()
        string = f"{string} [COL] description [VAL] {' '.join(desc.split(' ')[:100])}".strip()
        return string

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self.serialize(row)

        enc = self.tok(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = int(row['label'])
        return item

import torch
import torch.nn as nn
from transformers import BertModel

class BERT_EM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(cls)
        probs = self.sigmoid(logits).squeeze()

        if labels is not None:
            loss = nn.BCELoss()(probs, labels.float())
            return loss, probs

        return probs


tokenizer = BertTokenizer.from_pretrained(bert_model)

df = pd.read_csv("wdc_train_dataset.csv")

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"])

train_ds = Dataset(train_df, tokenizer)
val_ds = Dataset(val_df, tokenizer)
test_ds = Dataset(test_df, tokenizer)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)
test_loader = DataLoader(test_ds, batch_size=16)

from tqdm import tqdm
import torch.optim as optim

model = BERT_EM().cuda()
opt = optim.AdamW(model.parameters(), lr=2e-5)

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].cuda()
        mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()

        opt.zero_grad()
        loss, _ = model(input_ids, mask, labels)
        loss.backward()
        opt.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

from sklearn.metrics import f1_score, accuracy_score

def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].cuda()
            mask = batch["attention_mask"].cuda()
            labels = batch["labels"].numpy()

            probs = model(ids, mask).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels)

    preds = (np.array(all_probs) > 0.5).astype(int)
    f1 = f1_score(all_labels, preds)
    acc = accuracy_score(all_labels, preds)
    
    print("F1:", f1)
    print("Accuracy:", acc)
    return all_probs, all_labels

probs, labels = evaluate(model, test_loader)
