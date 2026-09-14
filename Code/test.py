import os
import torch
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
    "finetuned_dir": "../finetuned_models",
    "batch_size": 30,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

device = CONFIG["device"]

# =====================================================
# Evaluation function
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
    sn = tp / (tp + fn + 1e-8)
    sp = tn / (tn + fp + 1e-8)

    return {"acc": acc, "f1": f1, "mcc": mcc, "auc": auc, "sn": sn, "sp": sp}

# =====================================================
# Test single species
# =====================================================
def test_species(species, task_id):
    print("\n" + "="*80)
    print(f"Testing {species}")
    print("="*80)

    test_loader = get_single_task_loader(CONFIG["data_dir"], species, split="test", batch_size=CONFIG["batch_size"])

    model = RNAFoundationModel(num_tasks=12).to(device)

    model_path = os.path.join(CONFIG["finetuned_dir"], f"{species}_best.pth")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded Finetuned Model:", model_path)

    metrics = evaluate(model, test_loader, task_id)
    print("\nFinal Test Metrics for", species)
    print(
        f"ACC={metrics['acc']:.4f} | "
        f"AUC={metrics['auc']:.4f} | "
        f"MCC={metrics['mcc']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"SN={metrics['sn']:.4f} | "
        f"SP={metrics['sp']:.4f}"
    )
    return metrics

# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    TASKS = [
        "Atol", "hAm", "hCm", "hGm", "hm1A", "hm5C",
        "hm5U", "hm6A", "hm6Am", "hm7G", "hpsi", "hTm"
    ]

    all_results = {}

    for task_id, species in enumerate(TASKS):
        metrics = test_species(species, task_id)
        all_results[species] = metrics

    print("\n" + "="*80)
    print("FINAL SUMMARY OF ALL TASKS")
    print("="*80)
    for species, m in all_results.items():
        print(
            f"{species:8s} | "
            f"ACC={m['acc']:.4f} | "
            f"AUC={m['auc']:.4f} | "
            f"MCC={m['mcc']:.4f} | "
            f"F1={m['f1']:.4f} | "
            f"SN={m['sn']:.4f} | "
            f"SP={m['sp']:.4f}"
        )