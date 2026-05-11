import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from train_floorplan_detector import image_features, load_manifest


def predict_mask(artifact, image: Image.Image):
    image_size = int(artifact["image_size"])
    resized = image.convert("RGB").resize((image_size, image_size))
    feats = image_features(np.asarray(resized))
    flat = feats.reshape(-1, feats.shape[-1])
    model = artifact["model"]
    probabilities = model.predict_proba(flat)[:, 1].reshape(image_size, image_size)
    return probabilities


def main():
    parser = argparse.ArgumentParser(description="Evaluate the baseline room detector.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    rows = load_manifest(args.dataset, args.split)
    if not rows:
        raise SystemExit(f"No rows for split {args.split}.")

    all_truth = []
    all_pred = []
    image_size = int(artifact["image_size"])
    prediction_dir = args.output or args.model.parent / f"{args.split}_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows):
        image = Image.open(args.dataset / row["image"]).convert("RGB")
        mask = Image.open(args.dataset / row["mask"]).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
        truth = (np.asarray(mask) > 127).astype(np.uint8)
        pred = (predict_mask(artifact, image) >= 0.5).astype(np.uint8)
        all_truth.append(truth.reshape(-1))
        all_pred.append(pred.reshape(-1))
        if index < 16:
            Image.fromarray((pred * 255).astype(np.uint8)).save(prediction_dir / f"{row['sample_id']}_pred.png")

    truth = np.concatenate(all_truth)
    pred = np.concatenate(all_pred)
    metrics = {
        "split": args.split,
        "samples": len(rows),
        "pixel_accuracy": round(float(accuracy_score(truth, pred)), 4),
        "precision": round(float(precision_score(truth, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(truth, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(truth, pred, zero_division=0)), 4),
    }
    (prediction_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

