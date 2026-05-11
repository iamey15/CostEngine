import argparse
import json
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_manifest(dataset: Path, split: str):
    manifest = []
    with (dataset / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == split:
                manifest.append(row)
    return manifest


def image_features(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edges = cv2.Canny((gray * 255).astype(np.uint8), 60, 160).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx.astype(np.float32) / max(w - 1, 1)
    yy = yy.astype(np.float32) / max(h - 1, 1)
    return np.stack([gray, 1.0 - gray, edges, blur, np.abs(sobel_x), np.abs(sobel_y), xx, yy], axis=-1)


def sample_pixels(dataset: Path, rows: list, image_size: int, max_samples: int):
    per_image = max(500, max_samples // max(len(rows), 1))
    features = []
    labels = []
    rng = np.random.default_rng(42)
    for row in rows:
        image = Image.open(dataset / row["image"]).convert("RGB").resize((image_size, image_size))
        mask = Image.open(dataset / row["mask"]).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
        x = image_features(np.asarray(image))
        y = (np.asarray(mask) > 127).astype(np.uint8)
        positive = np.column_stack(np.where(y == 1))
        negative = np.column_stack(np.where(y == 0))
        half = per_image // 2
        if len(positive):
            selected = positive[rng.choice(len(positive), size=min(half, len(positive)), replace=len(positive) < half)]
            features.append(x[selected[:, 0], selected[:, 1]])
            labels.append(np.ones(len(selected), dtype=np.uint8))
        if len(negative):
            selected = negative[rng.choice(len(negative), size=min(half, len(negative)), replace=len(negative) < half)]
            features.append(x[selected[:, 0], selected[:, 1]])
            labels.append(np.zeros(len(selected), dtype=np.uint8))
    return np.vstack(features), np.concatenate(labels)


def predict_mask(model, image: Image.Image, image_size: int):
    resized = image.convert("RGB").resize((image_size, image_size))
    feats = image_features(np.asarray(resized))
    flat = feats.reshape(-1, feats.shape[-1])
    probabilities = model.predict_proba(flat)[:, 1].reshape(image_size, image_size)
    return probabilities


def evaluate_baseline(model, dataset: Path, rows: list, image_size: int, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    all_truth = []
    all_pred = []
    for index, row in enumerate(rows):
        image = Image.open(dataset / row["image"]).convert("RGB")
        mask = Image.open(dataset / row["mask"]).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
        truth = (np.asarray(mask) > 127).astype(np.uint8)
        prediction = (predict_mask(model, image, image_size) >= 0.5).astype(np.uint8)
        all_truth.append(truth.reshape(-1))
        all_pred.append(prediction.reshape(-1))
        if index < 8:
            Image.fromarray((prediction * 255).astype(np.uint8)).save(output / f"{row['sample_id']}_pred.png")
    truth = np.concatenate(all_truth)
    pred = np.concatenate(all_pred)
    return {
        "pixel_accuracy": round(float(accuracy_score(truth, pred)), 4),
        "precision": round(float(precision_score(truth, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(truth, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(truth, pred, zero_division=0)), 4),
        "samples": len(rows),
    }


def train_baseline(args):
    train_rows = load_manifest(args.dataset, "train")
    val_rows = load_manifest(args.dataset, "val")
    if not train_rows or not val_rows:
        raise SystemExit("Dataset must contain train and val splits.")

    x_train, y_train = sample_pixels(args.dataset, train_rows, args.image_size, args.max_samples)
    model = HistGradientBoostingClassifier(max_iter=args.iterations, learning_rate=0.08, max_leaf_nodes=31, random_state=42)
    model.fit(x_train, y_train)

    args.output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_baseline(model, args.dataset, val_rows, args.image_size, args.output / "validation_predictions")
    artifact = {
        "model": model,
        "image_size": args.image_size,
        "features": ["gray", "ink", "edges", "blur", "sobel_x_abs", "sobel_y_abs", "x_norm", "y_norm"],
        "type": "room_binary_patch_classifier",
    }
    joblib.dump(artifact, args.output / "model.joblib")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def train_yolo(args):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"Ultralytics is not installed. Install ml/floorplan_training/requirements-ml.txt first. {exc}")
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.yolo_base)
    result = model.train(data=str(args.dataset / "yolo_data.yaml"), epochs=args.epochs, imgsz=args.yolo_image_size, project=str(args.output), name="room-seg")
    print(result)


def main():
    parser = argparse.ArgumentParser(description="Train floorplan room detection models.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["baseline", "yolo"], default="baseline")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=220000)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--yolo-image-size", type=int, default=640)
    parser.add_argument("--yolo-base", default="yolov8n-seg.pt")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    if args.model == "baseline":
        train_baseline(args)
    else:
        train_yolo(args)


if __name__ == "__main__":
    main()

