import sys
import os
import json
from pathlib import Path

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.model.vlm_engine import vlm_engine
from backend.model.image_processor import SatelliteImageProcessor

def test_engine_on_vrsbench_sample():
    print("=" * 70)
    print("[SatAssist] Testing VLM Engine on Local VRSBench Data")
    print("=" * 70)

    # 1. Load a sample from local val dataset
    val_json_path = BASE_DIR / "data" / "vrsbench_5k" / "vrsbench_5k_val.json"
    if not val_json_path.exists():
        print(f"❌ Dataset file not found at: {val_json_path}")
        return

    with open(val_json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Pick the first sample with an existing image
    sample = None
    for s in samples:
        img_path = Path(s["image_path"])
        if img_path.exists():
            sample = s
            break

    if not sample:
        print("❌ No extracted image found on disk.")
        return

    img_path = Path(sample["image_path"])
    print(f"\n🖼️ Selected Test Image: {sample['image_file']} ({img_path})")
    if sample.get("caption"):
        print(f"📄 Ground Truth Caption:\n   \"{sample['caption']}\"")

    # 2. Test Image Processing
    print("\n🔍 Step 1: Testing Image Preprocessing...")
    img = SatelliteImageProcessor.load_image(img_path)
    meta = SatelliteImageProcessor.get_image_metadata(img)
    print(f"✅ Image loaded successfully! Dimensions: {meta['width']}x{meta['height']} ({meta['mode']})")

    # 3. Test Scene Captioning
    print("\n📝 Step 2: Testing Remote Sensing Captioning...")
    res_caption = vlm_engine.generate_caption(img_path)
    print(f"🤖 AI Description:\n   {res_caption['answer']}")

    # 4. Test Visual Question Answering (VQA)
    vqa_pairs = sample.get("vqa_pairs", [])
    test_question = vqa_pairs[0]["question"] if vqa_pairs else "What type of area is shown in this satellite image?"
    print(f"\n❓ Step 3: Testing VQA Question: \"{test_question}\"")
    res_vqa = vlm_engine.answer_question(img_path, test_question)
    print(f"🤖 AI Answer:\n   {res_vqa['answer']}")
    if res_vqa.get("bounding_boxes"):
        print(f"📍 Extracted Bounding Boxes: {res_vqa['bounding_boxes']}")

    # 5. Test Visual Grounding
    print("\n🎯 Step 4: Testing Visual Grounding Localization...")
    res_grounding = vlm_engine.visual_grounding(img_path, "buildings and main facilities")
    print(f"🤖 Localization Result: {res_grounding['answer']}")
    print(f"📍 Bounding Boxes: {res_grounding['bounding_boxes']}")

    print("\n" + "=" * 70)
    print("🎉 VLM Engine Test Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    test_engine_on_vrsbench_sample()
