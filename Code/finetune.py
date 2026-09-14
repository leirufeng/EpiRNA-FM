import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from datasets import get_single_task_loader

from models import RNAFoundationModel

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)


# =====================================================
# Config
# =====================================================

CONFIG = {
    "data_dir": "../datas",
    "pretrained_dir": "../pretrained_model",
    "save_dir": "../finetuned_models",
    "batch_size": 20,
    "epochs": 200,
    "patience": 35,
    "lr": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

os.makedirs(
    CONFIG["save_dir"],
    exist_ok=True
)

device = CONFIG["device"]


# =====================================================
# Metrics
# =====================================================

def evaluate(model, loader, task_id):
    model.eval()

    preds = []
    probs = []
    labels_all = []

    with torch.no_grad():
        for batch in loader:

            x1 = batch[0].to(device)
            x2 = batch[1].to(device)
            labels = batch[2].to(device)
            
            task_ids = torch.full((labels.size(0),), task_id, dtype=torch.long, device=device)

            outputs = model(x1, x2, task_ids)

            logits = outputs["logits"]

            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)

            probs.extend(prob.cpu().numpy())
            preds.extend(pred.cpu().numpy())

            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)
    mcc = matthews_corrcoef(labels_all, preds)
    auc = roc_auc_score(labels_all, probs)
    cm = confusion_matrix(labels_all, preds)

    tn, fp, fn, tp = cm.ravel()

    sn = tp/(tp+fn+1e-8)
    sp = tn/(tn+fp+1e-8)

    return {"acc": acc, "f1": f1, "mcc": mcc, "auc": auc, "sn": sn, "sp": sp}




def freeze_encoder(model):

    for name, param in model.encoder.named_parameters():
        param.requires_grad = False

    print("Encoder Frozen")

def unfreeze_last_blocks(model):

    for name, param in model.encoder.named_parameters():
        if "cross" in name:
            param.requires_grad = True

        if "cmse" in name:
            param.requires_grad = True

    print("CrossAttention Unfrozen")



def finetune_species(species, task_id):

    print("\n")
    print("="*80)
    print(f"Finetuning {species}")
    print("="*80)

    train_loader = get_single_task_loader(CONFIG["data_dir"], species, split="train", batch_size=CONFIG["batch_size"])

    valid_loader = get_single_task_loader(CONFIG["data_dir"], species, split="valid", batch_size=CONFIG["batch_size"])

    test_loader = get_single_task_loader(CONFIG["data_dir"], species, split="test", batch_size=CONFIG["batch_size"])

    # -------------------------------------------------
    # Load Foundation Model
    # -------------------------------------------------

    model = RNAFoundationModel(num_tasks=12).to(device)

    encoder_path = os.path.join(CONFIG["pretrained_dir"], "foundation_encoder.pth")

    model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    print("Loaded Foundation Encoder")

    freeze_encoder(model)

    # unfreeze_last_blocks(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr"])

    best_mcc = 0

    patience_counter = 0

    save_path = os.path.join(CONFIG["save_dir"], f"{species}_best.pth")

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    for epoch in range(CONFIG["epochs"]):
        model.train()

        running_loss = 0

        pbar = tqdm(train_loader)

        for batch in pbar:

            x1 = batch[0].to(device)
            x2 = batch[1].to(device)
            labels = batch[2].to(device)

            task_ids = torch.full((labels.size(0),), task_id, dtype=torch.long, device=device)

            optimizer.zero_grad()

            outputs = model(x1, x2, task_ids)

            logits = outputs["logits"]

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            pbar.set_description(f"Loss:{loss.item():.4f}")

        metrics = evaluate(model, valid_loader, task_id)

        print(
            f"\nEpoch {epoch+1}"
            f" ACC={metrics['acc']:.4f}"
            f" AUC={metrics['auc']:.4f}"
            f" MCC={metrics['mcc']:.4f}"
        )

        if metrics["mcc"] > best_mcc:
            best_mcc = metrics["mcc"]
            patience_counter = 0
            torch.save(model.state_dict(), save_path)

            print(f"Saved Best Model " f"AUC={best_mcc:.4f}")

        else:
            patience_counter += 1

        if patience_counter >= CONFIG["patience"]:
            print("Early Stop")

            break

    # -------------------------------------------------
    # Test
    # -------------------------------------------------

    print("\nLoading Best Model")

    model.load_state_dict(torch.load(save_path, map_location=device))

    metrics = evaluate(model, test_loader, task_id)

    print("\nFinal Test Results")

    print(metrics)

    return metrics


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    TASKS = [
        "Atol",
        "hAm",
        "hCm",
        "hGm",
        "hm1A",
        "hm5C",
        "hm5U",
        "hm6A",
        "hm6Am",
        "hm7G",
        "hpsi",
        "hTm"
    ]

    all_results = {}

    for task_id, species in enumerate(TASKS):
        metrics = finetune_species(species, task_id)
        all_results[species] = metrics

    print("\n")
    print("="*80)
    print("FINAL RESULTS")
    print("="*80)

    for species, m in all_results.items():
        print(
            f"{species:8s}"
            f" ACC={m['acc']:.4f}"
            f" AUC={m['auc']:.4f}"
            f" MCC={m['mcc']:.4f}"
            f" F1={m['f1']:.4f}"
            f" SN={m['sn']:.4f}"
            f" SP={m['sp']:.4f}"
        )