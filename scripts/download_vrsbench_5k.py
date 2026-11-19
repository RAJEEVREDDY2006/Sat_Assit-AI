"""
VRSBench 5,000 Subset Downloader & Curator with Seeded Random Sampling (Seed=42)
Author: Sat_Assist Team

This script:
1. Fetches official `VRSBench_train.json` from Hugging Face (`xiang709/VRSBench`).
2. Groups all multi-turn VQA, Captioning, and Visual Grounding conversations by unique image.
3. Uses reproducible seeded random sampling (Seed=42) to select 5,000 unique satellite images.
4. Streams/extracts only the 5,000 selected images from `Images_train.zip` directly into `data/vrsbench_5k/images/`.
5. Creates standardized 80/20 train/val splits:
   - `data/vrsbench_5k/vrsbench_5k_train.json` (4,000 images)
   - `data/vrsbench_5k/vrsbench_5k_val.json` (1,000 images)
   - `data/vrsbench_5k/dataset_stats.json`
"""

import os
import sys
import json
import random
import zipfile
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download

REPO_ID = "xiang709/VRSBench"

def setup_directories(base_dir: Path):
    data_dir = base_dir / "data" / "vrsbench_5k"
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, images_dir

def download_and_sample_vrsbench(
    target_count: int = 5000,
    seed: int = 42,
    train_ratio: float = 0.8,
    extract_images: bool = True
):
    print("=" * 65)
    print(f"🛰️  VRSBench 5k Downloader & Curator (Random Sampling, Seed={seed})")
    print("=" * 65)

    base_dir = Path(__file__).resolve().parent.parent
    data_dir, images_dir = setup_directories(base_dir)

    print("\n1️⃣ Downloading & loading VRSBench annotation metadata (VRSBench_train.json)...")
    annotation_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="VRSBench_train.json",
        repo_type="dataset"
    )

    with open(annotation_path, "r", encoding="utf-8") as f:
        raw_annotations = json.load(f)

    print(f"✅ Loaded {len(raw_annotations)} total conversation records.")

    # Group all conversation records by image filename
    image_to_records = {}
    for item in raw_annotations:
        img_name = item.get("image")
        if not img_name:
            continue
        if img_name not in image_to_records:
            image_to_records[img_name] = []
        image_to_records[img_name].append(item)

    unique_images = sorted(list(image_to_records.keys()))
    print(f"📊 Found {len(unique_images)} unique satellite images in the dataset.")

    # Seeded random sampling
    random.seed(seed)
    sampled_image_names = random.sample(unique_images, min(target_count, len(unique_images)))
    sampled_set = set(sampled_image_names)
    print(f"🎲 Sampled {len(sampled_image_names)} unique images with random seed={seed}.")

    # 2️⃣ Extract / Download Images
    if extract_images:
        print("\n2️⃣ Downloading & extracting selected 5,000 images from Images_train.zip...")
        zip_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="Images_train.zip",
            repo_type="dataset"
        )
        print(f"📦 Archive cached at: {zip_path}")
        print("📂 Extracting selected 5,000 images...")
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            # Map bare filename to archive path
            archive_map = {Path(p).name: p for p in namelist if not p.endswith("/")}
            
            extracted_count = 0
            for img_name in tqdm(sampled_image_names, desc="Extracting images"):
                target_file = images_dir / img_name
                if target_file.exists() and target_file.stat().st_size > 0:
                    extracted_count += 1
                    continue
                
                if img_name in archive_map:
                    with zf.open(archive_map[img_name]) as src, open(target_file, "wb") as dst:
                        dst.write(src.read())
                    extracted_count += 1
            print(f"✅ Extracted {extracted_count} satellite images to {images_dir}")

    # 3️⃣ Format and Split Annotations
    print("\n3️⃣ Formatting dataset samples & splits...")
    structured_samples = []
    
    for img_name in sampled_image_names:
        records = image_to_records[img_name]
        
        # Parse captions, vqa, and grounding
        captions = []
        vqa_list = []
        conversations = []
        
        for r in records:
            conv = r.get("conversations", [])
            conversations.extend(conv)
            
            # Categorize human query
            if len(conv) >= 2:
                q_text = conv[0].get("value", "")
                a_text = conv[1].get("value", "")
                
                if "[caption]" in q_text:
                    captions.append(a_text)
                else:
                    vqa_list.append({
                        "question": q_text.replace("<image>\n", "").strip(),
                        "answer": a_text.strip()
                    })
        
        sample_entry = {
            "image_id": Path(img_name).stem,
            "image_file": img_name,
            "image_path": str(images_dir / img_name),
            "caption": captions[0] if captions else "",
            "all_captions": captions,
            "vqa_pairs": vqa_list,
            "total_qa_pairs": len(vqa_list),
            "conversations": conversations
        }
        structured_samples.append(sample_entry)

    # Split into 80% Train, 20% Val
    random.shuffle(structured_samples)
    train_size = int(len(structured_samples) * train_ratio)
    train_split = structured_samples[:train_size]
    val_split = structured_samples[train_size:]

    # Save to JSON
    train_file = data_dir / "vrsbench_5k_train.json"
    val_file = data_dir / "vrsbench_5k_val.json"
    all_file = data_dir / "vrsbench_5k_all.json"
    stats_file = data_dir / "dataset_stats.json"

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_split, f, indent=2, ensure_ascii=False)
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_split, f, indent=2, ensure_ascii=False)
    with open(all_file, "w", encoding="utf-8") as f:
        json.dump(structured_samples, f, indent=2, ensure_ascii=False)

    # Compute & Save Statistics
    stats = {
        "dataset_name": "VRSBench-5k-Curated",
        "random_seed": seed,
        "total_unique_images": len(structured_samples),
        "train_images": len(train_split),
        "val_images": len(val_split),
        "total_vqa_pairs": sum(s["total_qa_pairs"] for s in structured_samples),
        "total_captions": sum(len(s["all_captions"]) for s in structured_samples),
        "image_resolution": "512x512",
        "format": "JPG / PNG Remote Sensing Imagery"
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 65)
    print("🎉 VRSBench 5,000 Dataset Pipeline Ready!")
    print(f"📁 Images:     {images_dir}")
    print(f"📄 Train Set:  {train_file} ({len(train_split)} images)")
    print(f"📄 Val Set:    {val_file} ({len(val_split)} images)")
    print(f"📊 Stats:      {stats_file}")
    print(f"Total VQA pairs sampled: {stats['total_vqa_pairs']}")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and curate VRSBench dataset subset")
    parser.add_argument("--count", type=int, default=5000, help="Number of unique satellite images to sample (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio for train split (default: 0.8)")
    parser.add_argument("--no-extract", action="store_true", help="Skip downloading/extracting image binaries (metadata only)")
    
    args = parser.parse_args()
    download_and_sample_vrsbench(
        target_count=args.count,
        seed=args.seed,
        train_ratio=args.train_ratio,
        extract_images=not args.no_extract
    )
