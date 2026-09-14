import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from datasets import get_multitask_loader
from models import RNAFoundationModel
from task_graph import TaskGraphModule          
from contrastive import MultiTaskContrastiveLoss  


# =====================================================
# Config
# =====================================================

CONFIG = {
    "data_dir": "../datas",
    "batch_size": 512,
    "epochs": 200,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "save_dir": "../pretrained_model",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)
device = CONFIG["device"]


# =====================================================
# Species
# =====================================================

TASKS = [
    "Atol", "hAm", "hCm", "hGm", "hm1A", "hm5C",
    "hm5U", "hm6A", "hm6Am", "hm7G", "hpsi", "hTm"
]


# =====================================================
# Train
# =====================================================

def train_epoch(model, graph_module, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_ce = 0
    total_con = 0

    pbar = tqdm(loader)
    for batch in pbar:
        x1 = batch[0].to(device)
        x2 = batch[1].to(device)
        labels = batch[2].to(device)       
        task_ids = batch[3].to(device)     

        optimizer.zero_grad()

        # --------------------------------
        # Forward
        # --------------------------------
        outputs = model(x1, x2, task_ids)
        feat = outputs["feature"]
        logits = outputs["logits"]

        # --------------------------------
        # Update Prototype (now requires labels)
        # --------------------------------
        graph_out = graph_module(feat.detach(), task_ids, labels)   
        
        prototype_pos = graph_out["graph_proto"]      
        prototype_neg = graph_out["prototype_neg"]    
        # --------------------------------
        # Loss (now passes both positive and negative prototypes)
        # --------------------------------
        loss, ce_loss, con_loss = criterion(
            logits, labels, feat, task_ids,
            pos_prototype=prototype_pos,   
            neg_prototype=prototype_neg    
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_con += con_loss.item()

        pbar.set_description(
            f"Loss:{loss.item():.4f} CE:{ce_loss.item():.4f} CON:{con_loss.item():.4f}"
        )

    n = len(loader)
    return total_loss / n, total_ce / n, total_con / n


# =====================================================
# Main
# =====================================================

def main():
    print("=" * 80)
    print("Loading MultiTask Dataset")
    print("=" * 80)

    train_loader = get_multitask_loader(
        CONFIG["data_dir"], split="train", batch_size=CONFIG["batch_size"]
    )

    print("Building Foundation Model...")
    model = RNAFoundationModel(num_tasks=len(TASKS)).to(device)

    
    graph_module = TaskGraphModule(
        num_tasks=len(TASKS),
        feat_dim=384,
        prior_mode='hard_mask',   
        prior_alpha=0.5,          
        temperature=1.0           
    ).to(device)

    criterion = MultiTaskContrastiveLoss(temperature=0.07, alpha=0.1)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(graph_module.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    best_loss = 999

    print("=" * 80)
    print("Start Pretraining")
    print("=" * 80)

    for epoch in range(CONFIG["epochs"]):
        loss, ce, con = train_epoch(model, graph_module, train_loader, optimizer, criterion)

        print(
            f"\nEpoch {epoch+1}/{CONFIG['epochs']} "
            f"Loss={loss:.4f} CE={ce:.4f} CON={con:.4f}"
        )

        if loss < best_loss:
            best_loss = loss
            print("Saving best model...")
            torch.save(
                model.encoder.state_dict(),
                os.path.join(CONFIG["save_dir"], "foundation_encoder.pth")
            )
            torch.save(
                graph_module.state_dict(),
                os.path.join(CONFIG["save_dir"], "foundation_graph.pth")
            )
            torch.save(
                model.state_dict(),
                os.path.join(CONFIG["save_dir"], "foundation_full.pth")
            )

    print("=" * 80)
    print("Pretraining Finished")
    print("=" * 80)

    # --------------------------------
    # Save final task relation graph
    # --------------------------------
    print("Saving final task relation graph...")
    with torch.no_grad():
        final_graph = graph_module()   
        adjacency = final_graph["adjacency"].cpu().numpy()
        graph_proto = final_graph["graph_proto"].cpu().numpy()
        print("adjacency shape:", adjacency)

    np.save(os.path.join(CONFIG["save_dir"], "final_adjacency.npy"), adjacency)
    np.save(os.path.join(CONFIG["save_dir"], "final_graph_proto.npy"), graph_proto)

    
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(adjacency, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.title('Task Relation Graph')
        plt.xlabel('Task')
        plt.ylabel('Task')
        plt.xticks(range(len(TASKS)), TASKS, rotation=45)
        plt.yticks(range(len(TASKS)), TASKS)
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["save_dir"], "task_relation_heatmap.png"))
        print("Heatmap saved.")
    except ImportError:
        print("matplotlib not installed, skip heatmap.")


if __name__ == "__main__":
    main()