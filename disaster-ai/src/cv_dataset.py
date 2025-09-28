# cv_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List

import json, random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---- Public label space (multiclass) ----
CLASSES  = ["no-damage", "minor-damage", "major-damage", "destroyed"]
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}

# ------------------------------------------------------------
# Helper: robustly find the GeoJSON label path for an image
# Supports:
#   <name>_label.json  OR  <name>.json
#   same directory as image OR a sibling ".../labels/" folder
# Works whether images live in ".../tier1/", ".../images/", etc.
# ------------------------------------------------------------
def _find_label_path(img_path: Path) -> Optional[Path]:
    """
    Locate the GeoJSON label for a post-disaster image.

    Tries:
      <name>_label.json
      <name>.json
      <name.replace('_post_disaster','')>_label.json
      <name.replace('_post_disaster','')>.json

    Searches same directory, sibling 'labels' if inside 'images',
    and a 'labels' folder directly under the current folder.
    """
    def candidates_for(base: str, root: Path):
        return [
            root / f"{base}_label.json",
            root / f"{base}.json",
        ]

    base = img_path.stem
    bases = {base}
    if "_post_disaster" in base:
        bases.add(base.replace("_post_disaster", ""))

    cand: list[Path] = []
    # A) same directory
    for b in bases:
        cand += candidates_for(b, img_path.parent)

    # B) sibling labels if we are in images/
    if img_path.parent.name == "images":
        lab = img_path.parent.parent / "labels"
        for b in bases:
            cand += candidates_for(b, lab)

    # C) defensive: 'labels' directly under current folder
    lab2 = img_path.parent / "labels"
    for b in bases:
        cand += candidates_for(b, lab2)

    for p in cand:
        if p.exists():
            return p
    return None



# ------------------------------------------------------------
# Dataset: 4-class building-damage classification from post-disaster tiles
# ------------------------------------------------------------
class XBDDamageDataset(Dataset):
    """
    Returns (image_tensor, label_id) where label_id ∈ {0..3} for
    ["no-damage", "minor-damage", "major-damage", "destroyed"].

    Folder `root` should contain a split (e.g., 'tier1' or 'train')
    with post-disaster PNG tiles. Labels are read from matching GeoJSONs.
    """
    def __init__(
        self,
        root: str | Path,
        split: str = "train",           # 'train' or 'val' (we 80/20 split)
        img_size: int = 224,
        seed: int = 42,
        limit: int | None = None
    ):
        self.root = Path(root)

        # Collect post-disaster images (defensive glob pattern covers both)
        items = sorted(self.root.rglob("*post*disaster*.png"))
        if not items:
            raise FileNotFoundError(
                f"No post-disaster PNGs under: {self.root} (looked for '*post*disaster*.png')"
            )

        # Deterministic shuffle + split 80/20
        rng = random.Random(seed)
        rng.shuffle(items)
        n_train = int(0.8 * len(items))
        self.items = items[:n_train] if split == "train" else items[n_train:]
        if limit is not None:
            self.items = self.items[:limit]

        # Image transforms
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.items)

    # --------- label reading (robust to malformed JSONs) ----------
    def _read_label(self, img_path: Path) -> int:
        label_json = _find_label_path(img_path)
        if label_json is None or label_json.suffix.lower() != ".json":
            return CLASS2ID["no-damage"]

        # Map various strings/numbers to our 4 classes
        ALIASES = {
            "no-damage": "no-damage",
            "no_damage": "no-damage",
            "none": "no-damage",
            "undamaged": "no-damage",
            "no": "no-damage",
            "0": "no-damage",

            "minor-damage": "minor-damage",
            "minor_damage": "minor-damage",
            "minor": "minor-damage",
            "1": "minor-damage",

            "major-damage": "major-damage",
            "major_damage": "major-damage",
            "major": "major-damage",
            "2": "major-damage",

            "destroyed": "destroyed",
            "destroy": "destroyed",
            "3": "destroyed",
        }

        def normalize_value(v) -> Optional[str]:
            # numbers like 0,1,2,3
            if isinstance(v, (int, float)):
                v = str(int(v))
            if not isinstance(v, str):
                return None
            s = v.strip().lower().replace(" ", "-").replace("_", "-")
            return ALIASES.get(s, s)

        # Try multiple possible property keys that appear in xBD variants
        CAND_KEYS = [
            "subtype", "damage", "damage_grade", "damage_type", "damage_level",
            "label", "category", "class", "status"
        ]

        try:
            raw = label_json.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, dict):
                return CLASS2ID["no-damage"]
            feats = data.get("features", [])
            if not isinstance(feats, list):
                return CLASS2ID["no-damage"]
        except Exception:
            return CLASS2ID["no-damage"]

        counts = [0, 0, 0, 0]
        for feat in feats:
            props = feat.get("properties", {}) if isinstance(feat, dict) else {}
            dmg_val = None
            # search all candidate keys
            for k in CAND_KEYS:
                if k in props and props[k] is not None:
                    dmg_val = normalize_value(props[k])
                    break
            # nothing found: keep going
            if dmg_val in CLASS2ID:
                counts[CLASS2ID[dmg_val]] += 1

        return int(counts.index(max(counts))) if sum(counts) > 0 else CLASS2ID["no-damage"]


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(self._read_label(img_path), dtype=torch.long)
        return x, y


# ------------------------------------------------------------
# Convenience: build train/val dataloaders from one split folder
# ------------------------------------------------------------
def create_dataloaders(
    data_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
    limit: int | None = None
) -> Tuple[DataLoader, DataLoader]:
    ds_tr = XBDDamageDataset(data_dir, split="train", img_size=img_size, seed=seed, limit=limit)
    ds_va = XBDDamageDataset(data_dir, split="val",   img_size=img_size, seed=seed, limit=limit)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl_tr, dl_va
