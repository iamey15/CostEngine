# Floorplan ML Training Workflow

This workspace is for improving flat-plan OCR and room detection without making the FastAPI demo heavier.

The workflow is intentionally active-learning first:

1. Pull at least 100 open-source floorplan samples with room polygons.
2. Train a fast baseline detector locally.
3. Run inference on new uploads.
4. Let the user correct zones in the canvas.
5. Export corrected zones back into the dataset.
6. Fine-tune again with the corrected samples mixed in.

## Quick Start

From the repo root:

```powershell
$py="C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py ml/floorplan_training/scripts/prepare_dataset.py --source hf --limit 100 --output ml/floorplan_training/data/floorplan_100
& $py ml/floorplan_training/scripts/train_floorplan_detector.py --dataset ml/floorplan_training/data/floorplan_100 --output ml/floorplan_training/artifacts/baseline_room_detector
& $py ml/floorplan_training/scripts/evaluate_floorplan_detector.py --dataset ml/floorplan_training/data/floorplan_100 --model ml/floorplan_training/artifacts/baseline_room_detector/model.joblib
```

If Hugging Face is unreachable, use:

```powershell
& $py ml/floorplan_training/scripts/prepare_dataset.py --source synthetic --limit 100 --output ml/floorplan_training/data/floorplan_100
```

## What Gets Produced

```text
ml/floorplan_training/data/floorplan_100/
  images/train|val|test/*.jpg
  masks/train|val|test/*.png
  labels/train|val|test/*.txt        # YOLO segmentation format
  annotations.json                   # COCO-style room polygons
  manifest.jsonl                     # compact metadata per sample
  yolo_data.yaml                     # optional Ultralytics training config

ml/floorplan_training/artifacts/baseline_room_detector/
  model.joblib
  metrics.json
  validation_predictions/
```

## Optional YOLO Segmentation Training

Install the ML-only dependencies:

```powershell
pip install -r ml/floorplan_training/requirements-ml.txt
```

Then:

```powershell
python ml/floorplan_training/scripts/train_floorplan_detector.py `
  --dataset ml/floorplan_training/data/floorplan_100 `
  --output ml/floorplan_training/artifacts/yolo_room_detector `
  --model yolo `
  --epochs 25
```

## LLM OCR Validation

The LLM validator sends compact JSON only. It does not send the full image.

```powershell
$env:LLM_VALIDATOR_API_KEY="your-key"
$env:LLM_VALIDATOR_API_URL="https://api-inference.huggingface.co/models/meta-llama/Llama-3-70B-Instruct"
python ml/floorplan_training/scripts/validate_ocr_with_llm.py --input sample_detection.json
```

The validator is meant to answer:

- Is the OCR output internally consistent?
- Are room counts suspicious?
- Are dimension strings malformed?
- Which labels should be corrected?

It should not replace geometry detection. It only validates uncertain OCR/detection results.

