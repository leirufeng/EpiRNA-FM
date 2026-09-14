import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# =====================================================
# Task Mapping
# =====================================================

TASKS = [
    'Atol',
    'hAm',
    'hCm',
    'hGm',
    'hm1A',
    'hm5C',
    'hm5U',
    'hm6A',
    'hm6Am',
    'hm7G',
    'hpsi',
    'hTm'
]

TASK2ID = {
    task:i
    for i,task in enumerate(TASKS)
}

ID2TASK = {
    i:task
    for i,task in enumerate(TASKS)
}


# =====================================================
# Base Dataset
# =====================================================

class RNADataset(Dataset):
    def __init__(self, sequences, labels, task_ids):

        self.sequences = sequences
        self.labels = labels
        self.task_ids = task_ids
        self.vocab = {'A':0, 'C':1, 'G':2, 'T':3, 'U':3, 'N':4}

    def __len__(self):

        return len(self.sequences)

    # =================================================
    # Crop
    # =================================================
    def crop_sequence(self, seq, center=501, flank=50):

        if len(seq) < 101:
            return seq

        center_idx = center - 1
        start = center_idx - flank
        end = center_idx + flank + 1

        return seq[start:end]

    # =================================================
    # OneHot
    # =================================================

    def one_hot(self, seq):

        mapping = {
            'A':[1,0,0,0],
            'C':[0,1,0,0],
            'G':[0,0,1,0],
            'T':[0,0,0,1],
            'U':[0,0,0,1],
            'N':[0.25,0.25,0.25,0.25]
        }

        return np.array(
            [mapping.get(x, [0.25] * 4) for x in seq],
            dtype=np.float32
        )

    # =================================================
    # Token
    # =================================================

    def encode(self, seq):

        return np.array(
            [self.vocab.get(s, 4) for s in seq],
            dtype=np.int64
        )

    def __getitem__(self, idx):

        seq = self.sequences[idx]
        seq = self.crop_sequence(seq)
        label = self.labels[idx]
        task_id = self.task_ids[idx]
        onehot = self.one_hot(seq).T
        token = self.encode(seq)

        return (
            torch.FloatTensor(onehot),
            torch.LongTensor(token),
            torch.LongTensor([label]),
            torch.LongTensor([task_id])
        )

        # return {
        #     "onehot": torch.FloatTensor(onehot),
        #     "token":  torch.LongTensor(token),
        #     "label": torch.LongTensor([label]),
        #     "task_id": torch.LongTensor([task_id])
        # }



# =====================================================
# Utils
# =====================================================

def load_txt(file):

    with open(file) as f:
        seqs = [
            line.strip().upper()
            for line in f
            if line.strip()
        ]

    return seqs


# =====================================================
# MultiTask Dataset
# =====================================================

class MultiTaskDataset(RNADataset):

    def __init__(self, data_dir, split):
        sequences = []
        labels = []
        task_ids = []
        for task in TASKS:
            pos_file = os.path.join(
                data_dir,
                split,
                f"{task}_{split}_pos.txt"
            )

            neg_file = os.path.join(
                data_dir,
                split,
                f"{task}_{split}_neg.txt"
            )

            if not os.path.exists(pos_file):
                continue

            if not os.path.exists(neg_file):
                continue

            pos = load_txt(pos_file)
            neg = load_txt(neg_file)
            sequences.extend(pos)

            labels.extend([1] * len(pos))
            task_ids.extend([TASK2ID[task]] * len(pos))
            sequences.extend(neg)
            labels.extend([0] * len(neg))
            task_ids.extend([TASK2ID[task]] * len(neg))

        super().__init__(sequences, labels, task_ids)

        print(f"\nLoaded {split}")
        print(f"Total Samples = {len(self)}")


# =====================================================
# Single Task Dataset
# =====================================================

class SingleTaskDataset(RNADataset):

    def __init__(self, data_dir, species, split):

        pos_file = os.path.join(
            data_dir,
            split,
            f"{species}_{split}_pos.txt"
        )

        neg_file = os.path.join(
            data_dir,
            split,
            f"{species}_{split}_neg.txt"
        )

        pos = load_txt(pos_file)
        neg = load_txt(neg_file)

        sequences = (pos + neg)

        labels = ([1] * len(pos) + [0] * len(neg))

        task_ids = ([TASK2ID[species]] * len(sequences))

        super().__init__(sequences, labels, task_ids)

        print(f"{species}-{split}")

        print(f"Samples={len(self)}")


# =====================================================
# Collate
# =====================================================

def collate_fn(batch):

    x1 = torch.stack([b[0] for b in batch])

    x2 = torch.stack([b[1] for b in batch])

    labels = torch.cat([b[2] for b in batch])

    task_ids = torch.cat([b[3] for b in batch])

    return (x1, x2, labels, task_ids)


# =====================================================
# Dataloader
# =====================================================

def get_multitask_loader(data_dir, split, batch_size=128):

    dataset = MultiTaskDataset(data_dir, split)

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=4, pin_memory=True, collate_fn=collate_fn)


def get_single_task_loader(data_dir, species, split, batch_size=128):

    dataset = SingleTaskDataset(data_dir, species, split)

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=4, pin_memory=True, collate_fn=collate_fn)


# =====================================================
# Debug
# =====================================================

if __name__ == "__main__":

    loader = get_multitask_loader(data_dir="../datas", split="train", batch_size=16)

    for x1, x2, y, task_id in loader:

        print(
            "OneHot:",
            x1.shape
        )

        print(
            "Token:",
            x2.shape
        )

        print(
            "Label:",
            y.shape
        )

        print(
            "Label:",
            y
        )

        print(
            "Task:",
            task_id.shape
        )

        print(
            "Task:",
            task_id
        )

        break