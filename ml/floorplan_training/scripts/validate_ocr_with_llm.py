import argparse
import json
import os
import re
from pathlib import Path

import requests


def compact_payload(payload: dict):
    summary = payload.get("summary") or {}
    ocr = payload.get("ocr") or {}
    rooms = payload.get("rooms") or []
    labels = ocr.get("labels") or payload.get("ocr_labels") or []
    dimensions = (ocr.get("dimension_evidence") or {}).get("dimensions_ft") or payload.get("dimensions") or []
    return {
        "summary": {
            "room_count": summary.get("room_count"),
            "bedroom_count": summary.get("bedroom_count"),
            "bathroom_count": summary.get("bathroom_count"),
            "kitchen_count": summary.get("kitchen_count"),
            "built_up_area_sqft": summary.get("built_up_area_sqft"),
            "polygon_area_sqft": summary.get("polygon_area_sqft"),
            "dimension_source": summary.get("dimension_source"),
        },
        "ocr_labels": labels[:40],
        "dimensions_ft": dimensions[:40],
        "rooms": [
            {
                "label": room.get("label"),
                "type": room.get("type"),
                "area_sqft": room.get("area_sqft"),
                "confidence": room.get("confidence"),
            }
            for room in rooms[:40]
        ],
        "signals": payload.get("signals") or {},
    }


def deterministic_validate(payload: dict):
    summary = payload.get("summary") or {}
    labels = [str(label).upper() for label in payload.get("ocr_labels") or []]
    rooms = payload.get("rooms") or []
    dimensions = payload.get("dimensions_ft") or []
    issues = []
    corrections = []

    if int(summary.get("room_count") or 0) > max(8, len(rooms) + 3):
        issues.append("Detected room count is higher than the room-zone list suggests.")
    if int(summary.get("bathroom_count") or 0) > 4 and len([label for label in labels if "BATH" in label or "TOILET" in label]) < 3:
        issues.append("Bathroom count looks overestimated compared with OCR labels.")
    for label in labels:
        fixed = re.sub(r"\bBED\s*ROON\b", "BEDROOM", label)
        fixed = re.sub(r"\bKITC?H?EN\b", "KITCHEN", fixed)
        fixed = re.sub(r"\bTOI[L1I]ET\b", "TOILET", fixed)
        if fixed != label:
            corrections.append({"from": label, "to": fixed})
    if not dimensions:
        issues.append("No usable dimensions were detected; cost estimate should depend on confirmed area.")
    verdict = "review" if issues else "pass"
    return {"verdict": verdict, "issues": issues, "label_corrections": corrections, "llm_used": False}


def llm_validate(payload: dict):
    api_url = os.getenv("LLM_VALIDATOR_API_URL")
    api_key = os.getenv("LLM_VALIDATOR_API_KEY")
    if not api_url or not api_key:
        return deterministic_validate(payload)

    prompt = {
        "task": "Validate floorplan OCR/detection JSON. Return only JSON with verdict pass/review/fail, issues, label_corrections, and concise_reason.",
        "rules": [
            "Do not invent rooms not supported by labels or geometry.",
            "Correct obvious architectural OCR typos.",
            "Flag suspicious room counts, missing dimensions, and area mismatch.",
            "Keep output compact."
        ],
        "floorplan_detection": payload,
    }
    response = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"inputs": json.dumps(prompt), "parameters": {"max_new_tokens": 220, "temperature": 0.1}},
        timeout=45,
    )
    response.raise_for_status()
    text = response.text
    try:
        parsed = response.json()
        if isinstance(parsed, list) and parsed and "generated_text" in parsed[0]:
            text = parsed[0]["generated_text"]
        elif isinstance(parsed, dict) and "generated_text" in parsed:
            text = parsed["generated_text"]
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        fallback = deterministic_validate(payload)
        fallback["raw_llm_output"] = text[:500]
        return fallback
    output = json.loads(match.group(0))
    output["llm_used"] = True
    return output


def main():
    parser = argparse.ArgumentParser(description="Validate OCR output with compact JSON and an optional LLM key.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = compact_payload(json.loads(args.input.read_text(encoding="utf-8-sig")))
    result = llm_validate(payload)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
