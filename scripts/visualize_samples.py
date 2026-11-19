"""
Preview & Visualizer for Curated VRSBench 5k Subset
Author: Sat_Assist Team

Usage:
    python scripts/visualize_samples.py --num 6
"""

import json
import random
from pathlib import Path
from PIL import Image

def visualize_dataset_samples(num_samples: int = 4, split: str = "train"):
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "vrsbench_5k"
    json_file = data_dir / f"vrsbench_5k_{split}.json"
    
    if not json_file.exists():
        print(f"❌ Annotations not found at: {json_file}")
        print("Please run `python scripts/download_vrsbench_5k.py` first.")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print("=" * 70)
    print(f"🛰️  VRSBench Curated Dataset Preview ({split.upper()} Split)")
    print(f"📊 Total samples available in split: {len(samples)}")
    print("=" * 70)

    selected = random.sample(samples, min(num_samples, len(samples)))

    for i, s in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] Image: {s['image_file']} (ID: {s['image_id']})")
        print(f"📍 File Path: {s['image_path']}")
        
        # Check if image file exists
        img_exists = Path(s["image_path"]).exists()
        print(f"🖼️ Image on disk: {'✅ Found' if img_exists else '⏳ Not yet extracted'}")
        
        if s.get("caption"):
            print(f"📝 Caption:\n   \"{s['caption']}\"")
        
        vqa = s.get("vqa_pairs", [])
        if vqa:
            print(f"❓ Sample VQA Pairs ({len(vqa)} total):")
            for q in vqa[:3]:
                print(f"   Q: {q['question']}")
                print(f"   A: {q['answer']}")
        print("-" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=4, help="Number of random samples to display")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"])
    args = parser.parse_args()
    visualize_dataset_samples(num_samples=args.num, split=args.split)
