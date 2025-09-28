from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List
import json, random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Public label order you can reuse elsewhere
CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
CLASS2ID = {c:i for i,c in enumerate(CLASSES)}

def _find_label_path(img_path: Path) -> Optional[Path]:
    """
    Robustly locate the GeoJSON label corresponding to a post-disaster image.
    Handles both:
      A) .../some_dir/IMAGE_post_disaster.png
         .../some_dir/IMAGE_post_disaster_label.json
      B) .../images/IMAGE_post_disaster.png
         .../labels/IMAGE_post_disaster_label.json   (sibling to images/)
    """
    # 1) same directory
    candidate = img_path.with_suffix("").with_name(img_path.stem.replace(".png","") + "_label.json")
    if candidate.exists():
        return candidate

    # 2) sibling 'labels' directory
    if img_path.parent.name == "images":
        cand2 = img_path.parent.parent / "labels" / (img_path.stem.replace(".png","") + "_label.json")
        if cand2.exists():
            return cand2

    # 3) parent/labels (when images/labels are at same level)
    cand3 = img_path.parent / "labels" / (img_path.stem.replace(".png","") + "_label.json")
    if cand3.exists():
        return cand3

    return None


class XBDDamageDataset(Dataset):
    """
    Reads post-disaster tiles + GeoJSON labels and returns (image_tensor, label_id).
    Label space = 4 classes: no/minor/major/destroyed.

    Args:
        root: folder containing a split (e.g., 'disaster-ai/data/xbd/tier1')
        split: 'train' or 'val' (we do an 80/20 split inside this folder)
        img_size: square resize for CNNs
        seed: shuffle seed
        limit: cap the number of samples (useful for quick tests)
    """
    def __init__(self, root: str, split: str = "train", img_size: int = 224, seed: int = 42, limit: int | None = None):
        self.root = Path(root)
        # find all post-disaster images under this split folder
        self.items: List[Path] = sorted(self.root.rglob("*post*disaster*.png"))
        if len(self.items) == 0:
            raise FileNotFoundError(f"No post-disaster PNGs found under: {self.root} (looked for '*post*disaster*.png')")

        random.seed(seed); random.shuffle(self.items)
        n = int(0.8 * len(self.items))
        self.items = self.items[:n] if split == "train" else self.items[n:]
        if limit is not None:
            self.items = self.items[:limit]

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def _read_label(self, img_path: Path) -> int:
        label_json = _find_label_path(img_path)
        if label_json is None:
            # default to no-damage if truly missing
            return CLASS2ID["no-damage"]

        data = json.loads(label_json.read_text())
        counts = [0,0,0,0]  # no, minor, major, destroyed
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            dmg = props.get("subtype") or props.get("damage")
            if dmg in CLASS2ID:
                counts[CLASS2ID[dmg]] += 1

        return int(counts.index(max(counts))) if sum(counts) > 0 else CLASS2ID["no-damage"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.items[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(self._read_label(p), dtype=torch.long)
        return x, y


def create_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
    limit: int | None = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience helper to build train/val DataLoaders from a single split folder
    like '.../train' or '.../tier1'.
    """
    ds_tr = XBDDamageDataset(data_dir, split="train", img_size=img_size, seed=seed, limit=limit)
    ds_va = XBDDamageDataset(data_dir, split="val",   img_size=img_size, seed=seed, limit=limit)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl_tr, dl_va
