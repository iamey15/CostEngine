import argparse
import json
import math
import random
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "peaceAsh/floorplan-room-segmentation"
ROOM_CLASS_ID = 0


def ensure_dirs(root: Path):
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_for_index(index: int):
    bucket = index % 10
    if bucket == 8:
        return "val"
    if bucket == 9:
        return "test"
    return "train"


def normalize_polygon(points, width, height):
    return [
        max(0.0, min(1.0, float(points[i]) / width)) if i % 2 == 0 else max(0.0, min(1.0, float(points[i]) / height))
        for i in range(len(points))
    ]


def polygon_bounds(points):
    xs = [float(points[i]) for i in range(0, len(points), 2)]
    ys = [float(points[i]) for i in range(1, len(points), 2)]
    return min(xs), min(ys), max(xs), max(ys)


def write_yolo_label(path: Path, annotations, width: int, height: int):
    lines = []
    for annotation in annotations:
        for segmentation in annotation.get("segmentation") or []:
            if len(segmentation) < 6:
                continue
            normalized = normalize_polygon(segmentation, width, height)
            lines.append(" ".join([str(ROOM_CLASS_ID), *[f"{value:.6f}" for value in normalized]]))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_mask(path: Path, annotations, width: int, height: int):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for annotation in annotations:
        for segmentation in annotation.get("segmentation") or []:
            if len(segmentation) < 6:
                continue
            points = [(float(segmentation[i]), float(segmentation[i + 1])) for i in range(0, len(segmentation), 2)]
            draw.polygon(points, fill=255)
    mask.save(path)


def coco_annotation(image_id: int, annotation_id: int, source_annotation: dict):
    segmentation = source_annotation.get("segmentation") or []
    first = segmentation[0] if segmentation else []
    bbox = source_annotation.get("bbox") or list(polygon_bounds(first)) if first else [0, 0, 1, 1]
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": segmentation,
        "area": float(source_annotation.get("area") or 0),
        "bbox": [float(value) for value in bbox],
        "iscrowd": 0,
    }


def fetch_hf_rows(limit: int, page_size: int = 100):
    rows = []
    offset = 0
    while len(rows) < limit:
        length = min(page_size, limit - len(rows))
        params = {
            "dataset": HF_DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
        response = requests.get(HF_ROWS_URL, params=params, timeout=45)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("rows") or []
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        time.sleep(0.2)
    return rows[:limit]


def download_image(url: str, target: Path, retries: int = 3):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            target.write_bytes(response.content)
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1 + attempt)


def prepare_hf_dataset(output: Path, limit: int):
    rows = fetch_hf_rows(limit)
    if len(rows) < limit:
        raise RuntimeError(f"Only fetched {len(rows)} rows from Hugging Face; requested {limit}.")

    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "room", "supercategory": "zone"}]}
    manifest_lines = []
    annotation_id = 1

    for index, entry in enumerate(rows):
        row = entry["row"]
        split = split_for_index(index)
        sample_id = f"hf_{index + 1:04d}"
        width = int(row.get("width") or row.get("image", {}).get("width") or 1080)
        height = int(row.get("height") or row.get("image", {}).get("height") or 1080)
        image_url = row.get("image", {}).get("src")
        if not image_url:
            raise RuntimeError(f"Missing image URL for row {index}.")

        image_path = output / "images" / split / f"{sample_id}.jpg"
        mask_path = output / "masks" / split / f"{sample_id}.png"
        label_path = output / "labels" / split / f"{sample_id}.txt"
        download_image(image_url, image_path)
        annotations = row.get("annotations") or []
        write_mask(mask_path, annotations, width, height)
        write_yolo_label(label_path, annotations, width, height)

        image_id = index + 1
        coco["images"].append(
            {
                "id": image_id,
                "file_name": str(image_path.relative_to(output)).replace("\\", "/"),
                "width": width,
                "height": height,
                "source": HF_DATASET,
                "original_file_name": row.get("file_name"),
                "split": split,
            }
        )
        for source_annotation in annotations:
            coco["annotations"].append(coco_annotation(image_id, annotation_id, source_annotation))
            annotation_id += 1
        manifest_lines.append(
            {
                "sample_id": sample_id,
                "split": split,
                "image": str(image_path.relative_to(output)).replace("\\", "/"),
                "mask": str(mask_path.relative_to(output)).replace("\\", "/"),
                "label": str(label_path.relative_to(output)).replace("\\", "/"),
                "width": width,
                "height": height,
                "room_polygon_count": len(annotations),
                "source": HF_DATASET,
            }
        )
    return coco, manifest_lines


def random_room_grid(width: int, height: int):
    margin = random.randint(48, 78)
    cols = random.randint(2, 4)
    rows = random.randint(2, 3)
    grid_w = width - margin * 2
    grid_h = height - margin * 2
    xs = [margin]
    ys = [margin]
    for col in range(1, cols):
        xs.append(margin + int(grid_w * col / cols + random.randint(-24, 24)))
    for row in range(1, rows):
        ys.append(margin + int(grid_h * row / rows + random.randint(-24, 24)))
    xs.append(width - margin)
    ys.append(height - margin)
    rooms = []
    for row in range(rows):
        for col in range(cols):
            x1, x2 = xs[col], xs[col + 1]
            y1, y2 = ys[row], ys[row + 1]
            if random.random() < 0.12:
                continue
            inset = random.randint(3, 10)
            rooms.append([x1 + inset, y1 + inset, x2 - inset, y2 - inset])
    return rooms


def synthetic_annotations(rooms):
    annotations = []
    for room in rooms:
        x1, y1, x2, y2 = room
        jitter = random.randint(0, 10)
        polygon = [
            x1,
            y1 + random.randint(0, jitter),
            x2 - random.randint(0, jitter),
            y1,
            x2,
            y2 - random.randint(0, jitter),
            x1 + random.randint(0, jitter),
            y2,
        ]
        annotations.append(
            {
                "area": max(1, (x2 - x1) * (y2 - y1)),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "category_id": 1,
                "segmentation": [polygon],
            }
        )
    return annotations


def draw_synthetic_plan(path: Path, rooms, width: int, height: int):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    line_width = random.choice([5, 6, 7, 8])
    for room in rooms:
        x1, y1, x2, y2 = room
        draw.rectangle([x1, y1, x2, y2], outline="black", width=line_width)
        if random.random() < 0.65:
            label = random.choice(["BEDROOM", "KITCHEN", "LIVING", "TOILET", "HALL", "DINING"])
            draw.text((x1 + 12, y1 + 14), label, fill=random.choice(["black", "#444444"]))
        if random.random() < 0.45:
            draw.line([x1 + 16, y1 - 14, x2 - 16, y1 - 14], fill="#444444", width=1)
            draw.text((x1 + 20, y1 - 30), f"{random.randint(8, 18)}'-{random.choice([0, 3, 6, 9])}\"", fill="#222222")
    image.save(path, quality=92)


def prepare_synthetic_dataset(output: Path, limit: int):
    random.seed(42)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "room", "supercategory": "zone"}]}
    manifest_lines = []
    annotation_id = 1
    width = height = 768
    for index in range(limit):
        split = split_for_index(index)
        sample_id = f"synthetic_{index + 1:04d}"
        image_path = output / "images" / split / f"{sample_id}.jpg"
        mask_path = output / "masks" / split / f"{sample_id}.png"
        label_path = output / "labels" / split / f"{sample_id}.txt"
        rooms = random_room_grid(width, height)
        annotations = synthetic_annotations(rooms)
        draw_synthetic_plan(image_path, rooms, width, height)
        write_mask(mask_path, annotations, width, height)
        write_yolo_label(label_path, annotations, width, height)
        image_id = index + 1
        coco["images"].append(
            {
                "id": image_id,
                "file_name": str(image_path.relative_to(output)).replace("\\", "/"),
                "width": width,
                "height": height,
                "source": "synthetic-bootstrap",
                "split": split,
            }
        )
        for source_annotation in annotations:
            coco["annotations"].append(coco_annotation(image_id, annotation_id, source_annotation))
            annotation_id += 1
        manifest_lines.append(
            {
                "sample_id": sample_id,
                "split": split,
                "image": str(image_path.relative_to(output)).replace("\\", "/"),
                "mask": str(mask_path.relative_to(output)).replace("\\", "/"),
                "label": str(label_path.relative_to(output)).replace("\\", "/"),
                "width": width,
                "height": height,
                "room_polygon_count": len(annotations),
                "source": "synthetic-bootstrap",
            }
        )
    return coco, manifest_lines


def write_dataset_files(output: Path, coco: dict, manifest_lines: list):
    (output / "annotations.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_lines:
            handle.write(json.dumps(row) + "\n")
    (output / "yolo_data.yaml").write_text(
        "\n".join(
            [
                f"path: {output.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: room",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare a floorplan room segmentation dataset.")
    parser.add_argument("--source", choices=["hf", "synthetic"], default="hf")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.limit < 100:
        raise SystemExit("--limit must be at least 100 for the requested bootstrap workflow.")
    args.output.mkdir(parents=True, exist_ok=True)
    ensure_dirs(args.output)

    if args.source == "hf":
        try:
            coco, manifest_lines = prepare_hf_dataset(args.output, args.limit)
        except Exception as exc:
            print(f"Hugging Face download failed: {exc}")
            print("Falling back to synthetic bootstrap dataset.")
            coco, manifest_lines = prepare_synthetic_dataset(args.output, args.limit)
    else:
        coco, manifest_lines = prepare_synthetic_dataset(args.output, args.limit)

    write_dataset_files(args.output, coco, manifest_lines)
    summary = {
        "output": str(args.output),
        "samples": len(manifest_lines),
        "source": sorted({row["source"] for row in manifest_lines}),
        "splits": {split: sum(1 for row in manifest_lines if row["split"] == split) for split in ("train", "val", "test")},
        "annotations": len(coco["annotations"]),
    }
    (args.output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
