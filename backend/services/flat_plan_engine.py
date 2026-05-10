import base64
import json
import math
import os
import re
import struct
import uuid
from collections import deque
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional CV dependency guard
    cv2 = None

try:
    from PIL import Image
    from PIL import ImageEnhance, ImageFilter, ImageOps
except Exception:  # pragma: no cover - pillow is bundled in local/demo env
    Image = None
    ImageEnhance = None
    ImageFilter = None
    ImageOps = None

try:
    import pytesseract
except Exception:  # pragma: no cover - optional OCR dependency guard
    pytesseract = None

try:
    import easyocr
except Exception:  # pragma: no cover - optional OCR dependency guard
    easyocr = None

try:
    import ezdxf
except Exception:  # pragma: no cover - optional parser guard for deploys
    ezdxf = None

try:
    import fitz
except Exception:  # pragma: no cover - optional PDF vector/text guard
    fitz = None


UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads" / "plans"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PLAN_META_DIR = UPLOAD_DIR / "_meta"
PLAN_META_DIR.mkdir(parents=True, exist_ok=True)

PLAN_STORE = {}
EASYOCR_READER = None

MATERIAL_RATES = {
    "steel": 74,
    "cement": 410,
    "sand": 78,
    "tiles": 105,
    "paint": 18,
    "bricks": 10,
    "concrete": 6200,
}
FLAT_FINISH_FACTORS = {"Standard": 1.0, "Premium": 1.16, "Luxury": 1.32}
MATERIAL_WASTE_FACTORS = {
    "Steel": 0.05,
    "Cement": 0.04,
    "Sand": 0.06,
    "Tiles": 0.10,
    "Paint": 0.08,
    "Bricks": 0.05,
    "Concrete": 0.03,
}

TESSERACT_CMD = os.getenv("TESSERACT_CMD") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if pytesseract and Path(TESSERACT_CMD).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def _image_size(data: bytes):
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"\xff\xd8"):
        index = 2
        while index < len(data) - 9:
            marker = data[index : index + 2]
            if marker[0] != 0xFF:
                index += 1
                continue
            block_length = int.from_bytes(data[index + 2 : index + 4], "big")
            if marker[1] in {0xC0, 0xC2}:
                height = int.from_bytes(data[index + 5 : index + 7], "big")
                width = int.from_bytes(data[index + 7 : index + 9], "big")
                return width, height
            index += 2 + block_length
    return None


def _quality_from_file(file_name: str, data: bytes):
    size = _image_size(data)
    ext = Path(file_name).suffix.lower().replace(".", "")
    score = 72
    notes = []
    if size:
        width, height = size
        pixels = width * height
        if pixels >= 1_000_000:
            score += 12
            notes.append("image resolution is suitable for room boundary detection")
        elif pixels < 300_000:
            score -= 12
            notes.append("image resolution is low, so smaller labels may be missed")
        contrast_seed = len(set(data[: min(len(data), 4000)]))
        if contrast_seed > 140:
            score += 5
        else:
            score -= 5
            notes.append("image contrast appears limited")
    elif ext == "dxf":
        score += 10
        notes.append("CAD vector geometry improves boundary confidence")
    elif ext == "pdf":
        score -= 4
        notes.append("PDF extraction may miss embedded raster dimensions")
    else:
        score -= 8
    return {"score": max(45, min(94, score)), "size": size, "notes": notes}


AREA_VALUE_RE = re.compile(r"(\d{2,6}(?:\.\d+)?)\s*(?:SQFT|SQ\.FT|SQ\s*FT|SOFT)")
ROOM_AREA_TEXT_RE = re.compile(r"\b(?:BED(?:ROOM|RM)?|BATH(?:ROOM)?|TOILET|WC|KITCHEN|KIT|LIVING|HALL|DINING|SITTING|UTILITY|BALCONY|VERANDA|PORCH|STORE|STORAGE|SERV(?:ICE)?)\b")
ARCH_SYMBOL_ALIASES = {
    "∅": " DIA ",
    "Ø": " DIA ",
    "⌀": " DIA ",
    "±": " PLUSMINUS ",
    "＋／－": " PLUSMINUS ",
}


def _unique_area_values(values, tolerance=1.5):
    unique = []
    for value in values:
        numeric = float(value)
        if all(abs(numeric - existing) > tolerance for existing in unique):
            unique.append(numeric)
    return unique


def _area_summary_from_text_rows(text_rows):
    overall_area_values = []
    room_area_values = []
    for raw in text_rows or []:
        normalized = re.sub(r"\s+", " ", str(raw or "").upper()).strip()
        values = [float(match) for match in AREA_VALUE_RE.findall(normalized)]
        if not values:
            continue
        if ROOM_AREA_TEXT_RE.search(normalized):
            room_area_values.extend(values)
        else:
            overall_area_values.extend(values)
    unique_overall = _unique_area_values(overall_area_values)
    unique_room = _unique_area_values(room_area_values)
    inferred_total = sum(unique_room) if len(unique_room) >= 3 else None
    total_area = max(unique_overall) if unique_overall else inferred_total
    largest_room_area = max(unique_room) if unique_room else None
    return {
        "total_area": round(total_area, 1) if total_area else None,
        "room_area_values_sqft": [round(value, 1) for value in unique_room[:24]],
        "room_area_sum_sqft": round(sum(unique_room), 1) if unique_room else None,
        "largest_room_area_sqft": round(largest_room_area, 1) if largest_room_area else None,
    }


def _normalize_arch_text(text: str):
    normalized = str(text or "")
    for symbol, alias in ARCH_SYMBOL_ALIASES.items():
        normalized = normalized.replace(symbol, alias)
    normalized = normalized.replace("’", "'").replace("‘", "'").replace("”", '"').replace("“", '"')
    normalized = normalized.upper()
    replacements = {
        "BFDROOM": "BEDROOM",
        "BED ROOM": "BEDROOM",
        "BEDRM": "BEDROOM",
        "BATH ROOM": "BATHROOM",
        "TORET": "TOILET",
        "TOKET": "TOILET",
        "T0ILET": "TOILET",
        "K1TCHEN": "KITCHEN",
        "K1T": "KIT",
        "KTTCHEN": "KITCHEN",
        "L1VING": "LIVING",
        "UT1LITY": "UTILITY",
        "UTIL1TY": "UTILITY",
        "S0FT": "SQFT",
        "SOFT": "SQFT",
        "SQ FT": "SQFT",
        "SQ.FT": "SQFT",
    }
    for wrong, right in replacements.items():
        normalized = re.sub(rf"\b{re.escape(wrong)}\b", right, normalized)
    normalized = re.sub(r"(?<=\d)O(?=\d|['\"\-])", "0", normalized)
    normalized = re.sub(r"(?<=[\d'\-])O(?=\d)", "0", normalized)
    normalized = re.sub(r"(?<=\d)[IL](?=\d|['\"\-])", "1", normalized)
    normalized = re.sub(r"(?<=[\d'\-])[IL](?=\d)", "1", normalized)
    normalized = re.sub(r"\bS\s*Q\s*F\s*T\b", "SQFT", normalized)
    return normalized


def _pdf_plan_text_and_preview(path: Path):
    if not fitz:
        return {"text": "", "preview_data_url": None, "page_count": 0}
    try:
        doc = fitz.open(str(path))
    except Exception:
        return {"text": "", "preview_data_url": None, "page_count": 0}
    texts = []
    preview_data_url = None
    try:
        for page_index, page in enumerate(doc):
            text = page.get_text("text") or ""
            if text.strip():
                texts.append(text)
            if page_index == 0:
                pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
                png = pix.tobytes("png")
                preview_data_url = f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"
        return {"text": "\n".join(texts), "preview_data_url": preview_data_url, "page_count": len(doc)}
    finally:
        doc.close()


def _point_xy(value):
    try:
        if value is None:
            return None
        if hasattr(value, "x") and hasattr(value, "y"):
            return float(value.x), float(value.y)
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            return float(value[0]), float(value[1])
    except Exception:
        return None
    return None


def _dxf_entity_points(entity, kind):
    points = []
    try:
        if kind == "LINE":
            for attr in ("start", "end"):
                point = _point_xy(getattr(entity.dxf, attr, None))
                if point:
                    points.append(point)
        elif kind == "LWPOLYLINE" and hasattr(entity, "get_points"):
            for point in entity.get_points("xy"):
                xy = _point_xy(point)
                if xy:
                    points.append(xy)
        elif kind == "POLYLINE" and hasattr(entity, "points"):
            for point in entity.points():
                xy = _point_xy(point)
                if xy:
                    points.append(xy)
        elif kind in {"CIRCLE", "ARC"}:
            center = _point_xy(getattr(entity.dxf, "center", None))
            radius = float(getattr(entity.dxf, "radius", 0) or 0)
            if center and radius > 0:
                cx, cy = center
                points.extend([(cx - radius, cy - radius), (cx + radius, cy + radius)])
    except Exception:
        return points
    return points


def _geometry_bounds(points):
    if len(points or []) < 2:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    width_units = max_x - min_x
    height_units = max_y - min_y
    if width_units < 1 or height_units < 1:
        return None
    return {
        "min_x": round(min_x, 2),
        "max_x": round(max_x, 2),
        "min_y": round(min_y, 2),
        "max_y": round(max_y, 2),
        "width_units": round(width_units, 2),
        "height_units": round(height_units, 2),
    }


def _plan_meta_path(plan_id: str):
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "", str(plan_id or ""))
    return PLAN_META_DIR / f"{safe_id}.json"


def _persist_plan(plan: dict):
    if not plan or not plan.get("id"):
        return
    serializable = {}
    for key, value in plan.items():
        if key == "preview_data_url" or key.startswith("_"):
            continue
        serializable[key] = value
    try:
        _plan_meta_path(plan["id"]).write_text(json.dumps(serializable, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception:
        pass


def _recover_plan_from_disk(plan_id: str):
    meta_path = _plan_meta_path(plan_id)
    if meta_path.exists():
        try:
            plan = json.loads(meta_path.read_text(encoding="utf-8"))
            path = Path(plan.get("path", ""))
            if path.exists():
                PLAN_STORE[plan_id] = plan
                return plan
        except Exception:
            pass

    matches = sorted(UPLOAD_DIR.glob(f"{re.sub(r'[^A-Za-z0-9_-]+', '', str(plan_id))}.*"))
    matches = [path for path in matches if path.is_file()]
    if not matches:
        return None
    path = matches[0]
    ext = path.suffix.lower().replace(".", "") or "png"
    try:
        data = path.read_bytes()
    except Exception:
        return None
    pdf_meta = _pdf_plan_text_and_preview(path) if ext == "pdf" else {}
    plan = {
        "id": plan_id,
        "project_id": None,
        "file_name": path.name,
        "file_type": ext,
        "path": str(path),
        "quality": _quality_from_file(path.name, data),
        "dxf_meta": _parse_dxf(path) if ext == "dxf" else {},
        "pdf_meta": pdf_meta,
        "label_hints": _label_hints_from_text(f"{path.name} {pdf_meta.get('text', '')}"),
        "demo": False,
        "recovered_from_disk": True,
    }
    PLAN_STORE[plan_id] = plan
    _persist_plan(plan)
    return plan


def _get_plan(plan_id: str):
    plan = PLAN_STORE.get(plan_id) or _recover_plan_from_disk(plan_id)
    if not plan:
        raise KeyError("Plan not found")
    return plan


def _demo_rooms(area=1250):
    scale = math.sqrt(area / 1250)
    rooms = [
        ("living", "Living / Hall", 5, 5, 46, 32, 330, 0.88),
        ("bedroom", "Bedroom 1", 55, 5, 34, 28, 240, 0.86),
        ("bedroom", "Bedroom 2", 55, 39, 34, 27, 220, 0.82),
        ("kitchen", "Kitchen", 5, 43, 25, 23, 130, 0.84),
        ("bathroom", "Bath 1", 34, 43, 16, 18, 55, 0.77),
        ("bathroom", "Bath 2", 34, 5, 16, 17, 50, 0.74),
        ("balcony", "Utility / Balcony", 5, 70, 84, 16, 95, 0.71),
    ]
    output = []
    for idx, (room_type, label, x, y, width, height, room_area, confidence) in enumerate(rooms, start=1):
        output.append(
            {
                "id": f"room-{idx}",
                "type": room_type,
                "label": label,
                "x": round(x, 1),
                "y": round(y, 1),
                "width": round(width, 1),
                "height": round(height, 1),
                "area_sqft": round(room_area * scale, 1),
                "confidence": confidence,
                "polygon": [
                    [round(x, 1), round(y, 1)],
                    [round(x + width, 1), round(y, 1)],
                    [round(x + width, 1), round(y + height, 1)],
                    [round(x, 1), round(y + height, 1)],
                ],
            }
        )
    return output


def _room_area_default(room_type: str, base_area: float):
    ratios = {
        "living": 0.18,
        "bedroom": 0.12,
        "kitchen": 0.08,
        "bathroom": 0.045,
        "balcony": 0.06,
        "service": 0.04,
    }
    minimums = {"living": 120, "bedroom": 90, "kitchen": 55, "bathroom": 32, "balcony": 35, "service": 28}
    return max(minimums.get(room_type, 35), float(base_area) * ratios.get(room_type, 0.06))


def _calculated_room_label(room_type: str, count: int):
    return _room_label_for_type(room_type, count)


def _counts_from_hints_and_dxf(label_hints: dict | None, dxf: dict | None):
    label_hints = label_hints or {}
    dxf = dxf or {}
    return {
        "living": max(int(label_hints.get("halls") or 0), int(dxf.get("halls") or 0), 1 if (label_hints.get("has_label_hint") or dxf.get("labels")) else 0),
        "bedroom": max(int(label_hints.get("bedrooms") or 0), int(dxf.get("bedrooms") or 0)),
        "kitchen": max(int(label_hints.get("kitchens") or 0), int(dxf.get("kitchens") or 0)),
        "bathroom": max(int(label_hints.get("bathrooms") or 0), int(dxf.get("bathrooms") or 0)),
        "balcony": int(label_hints.get("outdoor_zones") or 0),
        "service": int(label_hints.get("service_zones") or 0) + int(label_hints.get("stair_zones") or 0),
    }


def _area_from_room_text(text: str):
    values = [float(match) for match in AREA_VALUE_RE.findall(_normalize_arch_text(text))]
    values = [value for value in values if 20 <= value <= 1500]
    return values[0] if values else None


def _calculated_rooms_from_counts(base_area: float, label_hints: dict | None = None, dxf: dict | None = None):
    counts = _counts_from_hints_and_dxf(label_hints, dxf)
    specs = []
    for room_type in ("living", "bedroom", "kitchen", "bathroom", "balcony", "service"):
        for _ in range(max(0, counts.get(room_type, 0))):
            specs.append(room_type)
    if not specs:
        return []

    cols = max(2, math.ceil(math.sqrt(len(specs) * 1.25)))
    rows = math.ceil(len(specs) / cols)
    margin_x = 5.0
    margin_y = 5.0
    gap = 2.2
    cell_w = (125 - margin_x * 2 - gap * (cols - 1)) / cols
    cell_h = (92 - margin_y * 2 - gap * (rows - 1)) / rows
    counters = {}
    rooms = []
    total_default_area = sum(_room_area_default(room_type, base_area) for room_type in specs) or 1
    target_area = float(base_area) * 0.82
    area_scale = target_area / total_default_area
    for index, room_type in enumerate(specs):
        row = index // cols
        col = index % cols
        counters[room_type] = counters.get(room_type, 0) + 1
        x = margin_x + col * (cell_w + gap)
        y = margin_y + row * (cell_h + gap)
        area = _room_area_default(room_type, base_area) * area_scale
        aspect = {"living": 1.45, "bedroom": 1.18, "kitchen": 1.08, "bathroom": 0.9, "balcony": 1.8, "service": 1.0}.get(room_type, 1.1)
        width = min(cell_w, max(cell_w * 0.62, math.sqrt(max(area, 1) / max(target_area, 1) * (125 * 92) * aspect)))
        height = min(cell_h, max(cell_h * 0.55, width / aspect))
        rooms.append({
            "id": f"calc-zone-{index + 1}",
            "type": room_type,
            "label": _calculated_room_label(room_type, counters[room_type]),
            "x": round(x, 1),
            "y": round(y, 1),
            "width": round(width, 1),
            "height": round(height, 1),
            "area_sqft": round(area, 1),
            "confidence": 0.54,
            "polygon": [[round(x, 1), round(y, 1)], [round(x + width, 1), round(y, 1)], [round(x + width, 1), round(y + height, 1)], [round(x, 1), round(y + height, 1)]],
            "source": "calculated-count-layout",
        })
    return rooms


def _rooms_from_dxf_text_entities(dxf: dict, base_area: float, label_hints: dict | None = None):
    entities = dxf.get("text_entities") or []
    bounds = dxf.get("geometry_bounds") or {}
    width_units = float(bounds.get("width_units") or 0)
    height_units = float(bounds.get("height_units") or 0)
    if not entities or width_units <= 0 or height_units <= 0:
        return _calculated_rooms_from_counts(base_area, label_hints, dxf)

    min_x = float(bounds.get("min_x") or min(item["x"] for item in entities))
    min_y = float(bounds.get("min_y") or min(item["y"] for item in entities))
    counters = {}
    rooms = []
    for item in entities:
        canonical = _canonical_room_label(item.get("text"))
        if not canonical:
            continue
        room_type = canonical["type"]
        counters[room_type] = counters.get(room_type, 0) + 1
        x = max(3, min(118, ((float(item["x"]) - min_x) / width_units) * 112 + 6))
        y = max(3, min(85, 88 - ((float(item["y"]) - min_y) / height_units) * 80))
        area = _area_from_room_text(item.get("text")) or _room_area_default(room_type, base_area)
        aspect = {"living": 1.45, "bedroom": 1.18, "kitchen": 1.08, "bathroom": 0.9, "balcony": 1.8, "service": 1.0}.get(room_type, 1.1)
        width = max(7, min(24, math.sqrt(area / max(float(base_area), 1) * (125 * 92) * aspect)))
        height = max(6, min(18, width / aspect))
        rooms.append({
            "id": f"dxf-zone-{len(rooms) + 1}",
            "type": room_type,
            "label": _calculated_room_label(room_type, counters[room_type]),
            "x": round(max(0, min(125 - width, x - width / 2)), 1),
            "y": round(max(0, min(92 - height, y - height / 2)), 1),
            "width": round(width, 1),
            "height": round(height, 1),
            "area_sqft": round(area, 1),
            "confidence": 0.72,
            "source": "dxf-text-position",
        })
        room = rooms[-1]
        room["polygon"] = [[room["x"], room["y"]], [round(room["x"] + room["width"], 1), room["y"]], [round(room["x"] + room["width"], 1), round(room["y"] + room["height"], 1)], [room["x"], round(room["y"] + room["height"], 1)]]
    return rooms or _calculated_rooms_from_counts(base_area, label_hints, dxf)


def _parse_dxf(path: Path):
    if not ezdxf:
        return {}
    try:
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        texts = []
        text_entities = []
        line_count = 0
        vector_points = []
        for entity in msp:
            kind = entity.dxftype()
            if kind in {"LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC"}:
                vector_points.extend(_dxf_entity_points(entity, kind))
            if kind == "LINE":
                line_count += 1
            elif kind in {"LWPOLYLINE", "POLYLINE"}:
                line_count += 4
            elif kind in {"TEXT", "MTEXT"}:
                text = entity.plain_text() if kind == "MTEXT" else entity.dxf.text
                if text:
                    clean_text = _normalize_arch_text(text)
                    texts.append(clean_text)
                    insert = _point_xy(getattr(entity.dxf, "insert", None))
                    if insert:
                        text_entities.append({"text": clean_text, "x": round(insert[0], 3), "y": round(insert[1], 3)})
        joined = " ".join(texts)
        area_summary = _area_summary_from_text_rows(texts)
        return {
            "labels": texts,
            "text_entities": text_entities,
            "line_count": line_count,
            "has_dimensions": any(token in joined for token in ["SQFT", "SQ.FT", "FT", "AREA"]),
            "total_area": area_summary.get("total_area"),
            "room_area_values_sqft": area_summary.get("room_area_values_sqft"),
            "room_area_sum_sqft": area_summary.get("room_area_sum_sqft"),
            "largest_room_area_sqft": area_summary.get("largest_room_area_sqft"),
            "geometry_bounds": _geometry_bounds(vector_points),
            "text_dimension_evidence": _extract_dimension_evidence(joined, area_summary.get("total_area")),
            "bedrooms": max(1, joined.count("BED")),
            "bathrooms": joined.count("BATH") + joined.count("TOILET"),
            "kitchens": max(1 if "KITCHEN" in joined else 0, joined.count("KIT")),
            "halls": 1 if "LIVING" in joined or "HALL" in joined else 0,
        }
    except Exception:
        return {"parse_warning": "DXF parser could not read every entity, so heuristic mode was used."}


def register_plan(file_name: str, data: bytes, project_id: int, demo: bool = False):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(file_name).suffix.lower().replace(".", "") or "png"
    plan_id = f"plan-{uuid.uuid4().hex[:10]}"
    path = UPLOAD_DIR / f"{plan_id}.{ext}"
    path.write_bytes(data)
    quality = _quality_from_file(file_name, data)
    dxf_meta = _parse_dxf(path) if ext == "dxf" else {}
    pdf_meta = _pdf_plan_text_and_preview(path) if ext == "pdf" else {}
    label_hints = _label_hints_from_text(f"{file_name} {pdf_meta.get('text') or ''}")
    preview_data_url = None
    if ext in {"png", "jpg", "jpeg"}:
        mime = "image/png" if ext == "png" else "image/jpeg"
        preview_data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    elif ext == "pdf":
        preview_data_url = pdf_meta.get("preview_data_url")
    plan = {
        "id": plan_id,
        "project_id": project_id,
        "file_name": file_name,
        "file_type": ext,
        "path": str(path),
        "quality": quality,
        "dxf_meta": dxf_meta,
        "pdf_meta": pdf_meta,
        "label_hints": label_hints,
        "demo": demo,
    }
    PLAN_STORE[plan_id] = plan
    _persist_plan(plan)
    return {
        "plan_id": plan_id,
        "file_name": file_name,
        "file_type": ext,
        "image_size": quality.get("size"),
        "quality_score": quality["score"],
        "preview_data_url": preview_data_url,
        "message": "Floorplan uploaded and ready for AI-assisted layout processing.",
    }


def register_demo_plan(project_id: int):
    dxf = """0
SECTION
2
ENTITIES
0
TEXT
8
ROOM_LABELS
10
10
20
10
40
2.5
1
LIVING 330 SQFT
0
TEXT
8
ROOM_LABELS
10
55
20
10
40
2.5
1
BEDROOM 1
0
TEXT
8
ROOM_LABELS
10
55
20
42
40
2.5
1
BEDROOM 2
0
TEXT
8
ROOM_LABELS
10
12
20
48
40
2.5
1
KITCHEN
0
TEXT
8
ROOM_LABELS
10
35
20
50
40
2.5
1
BATH
0
ENDSEC
0
EOF
"""
    return register_plan("demo-flat-layout.dxf", dxf.encode("utf-8"), project_id, demo=True)


def _binary_dilate(mask, iterations=1):
    result = mask
    for _ in range(max(0, iterations)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
    return result


def _connected_components(mask, min_pixels):
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            queue = deque([(x, y)])
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            count = 0
            while queue:
                cx, cy = queue.popleft()
                count += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            if count >= min_pixels:
                components.append({"pixels": count, "bbox": (min_x, min_y, max_x, max_y)})
    return components


def _merge_bands(indices, max_gap=2):
    if not indices:
        return []
    bands = []
    start = prev = indices[0]
    for index in indices[1:]:
        if index - prev <= max_gap:
            prev = index
        else:
            bands.append((start, prev))
            start = prev = index
    bands.append((start, prev))
    return bands


def _classify_boxes_as_rooms(boxes, base_area, width, height, source="grid"):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda item: item["pixels"], reverse=True)[:16]
    total_pixels = sum(item["pixels"] for item in boxes) or 1
    area_ratios = [box["pixels"] / total_pixels for box in boxes]
    sorted_indices_by_size = sorted(range(len(boxes)), key=lambda idx: area_ratios[idx], reverse=True)
    largest_index = sorted_indices_by_size[0] if sorted_indices_by_size else None
    small_indices = [idx for idx in sorted_indices_by_size[::-1] if area_ratios[idx] < 0.11]
    bathroom_indices = set(small_indices[: min(2, len(small_indices))])
    kitchen_index = None
    for idx in sorted_indices_by_size[::-1]:
        if idx not in bathroom_indices and 0.07 <= area_ratios[idx] < 0.19:
            kitchen_index = idx
            break

    rooms = []
    for zero_index, box in enumerate(boxes):
        index = zero_index + 1
        min_x, min_y, max_x, max_y = box["bbox"]
        bbox_width = max_x - min_x + 1
        bbox_height = max_y - min_y + 1
        area_ratio = box["pixels"] / total_pixels
        room_area = max(30, round(base_area * 0.82 * area_ratio, 1))
        if zero_index == largest_index and area_ratio > 0.18:
            room_type = "living"
            label = "Living / Hall"
        elif zero_index in bathroom_indices:
            room_type = "bathroom"
            label = f"Bath / Toilet {sum(1 for room in rooms if room['type'] == 'bathroom') + 1}"
        elif zero_index == kitchen_index:
            room_type = "kitchen"
            label = "Kitchen"
        elif area_ratio < 0.055:
            room_type = "service"
            label = f"Small Zone {sum(1 for room in rooms if room['type'] == 'service') + 1}"
        else:
            room_type = "bedroom"
            label = f"Bedroom {sum(1 for room in rooms if room['type'] == 'bedroom') + 1}"
        rooms.append(
            {
                "id": f"room-{index}",
                "type": room_type,
                "label": label,
                "x": round(min_x / width * 125, 1),
                "y": round(min_y / height * 92, 1),
                "width": round(bbox_width / width * 125, 1),
                "height": round(bbox_height / height * 92, 1),
                "area_sqft": room_area,
                "confidence": round(max(0.36, min(0.76, 0.42 + area_ratio * 0.78)), 2),
                "polygon": [
                    [round(min_x / width * 125, 1), round(min_y / height * 92, 1)],
                    [round(max_x / width * 125, 1), round(min_y / height * 92, 1)],
                    [round(max_x / width * 125, 1), round(max_y / height * 92, 1)],
                    [round(min_x / width * 125, 1), round(max_y / height * 92, 1)],
                ],
                "source": source,
            }
        )
    total_area = sum(float(room["area_sqft"]) for room in rooms)
    target_area = max(1, float(base_area) * 0.82)
    if total_area:
        scale = target_area / total_area
        for room in rooms:
            room["area_sqft"] = round(float(room["area_sqft"]) * scale, 1)
    return rooms


def _line_projection_rooms(dark, base_area):
    height, width = dark.shape
    col_density = dark.mean(axis=0)
    row_density = dark.mean(axis=1)
    col_threshold = max(0.10, float(np.percentile(col_density, 88)))
    row_threshold = max(0.10, float(np.percentile(row_density, 88)))
    v_bands = _merge_bands([idx for idx, value in enumerate(col_density) if value >= col_threshold], max_gap=max(2, width // 160))
    h_bands = _merge_bands([idx for idx, value in enumerate(row_density) if value >= row_threshold], max_gap=max(2, height // 160))
    v_lines = [round((start + end) / 2) for start, end in v_bands if end - start >= 1]
    h_lines = [round((start + end) / 2) for start, end in h_bands if end - start >= 1]
    if not v_lines or v_lines[0] > width * 0.04:
        v_lines = [0] + v_lines
    if not v_lines or v_lines[-1] < width * 0.96:
        v_lines.append(width - 1)
    if not h_lines or h_lines[0] > height * 0.04:
        h_lines = [0] + h_lines
    if not h_lines or h_lines[-1] < height * 0.96:
        h_lines.append(height - 1)
    v_lines = sorted(set(v_lines))
    h_lines = sorted(set(h_lines))
    if len(v_lines) < 3 or len(h_lines) < 3 or len(v_lines) > 18 or len(h_lines) > 18:
        return []

    boxes = []
    min_area = width * height * 0.008
    for x1, x2 in zip(v_lines, v_lines[1:]):
        for y1, y2 in zip(h_lines, h_lines[1:]):
            cell_w = x2 - x1
            cell_h = y2 - y1
            if cell_w * cell_h < min_area:
                continue
            pad_x = max(2, round(cell_w * 0.08))
            pad_y = max(2, round(cell_h * 0.08))
            inner = dark[y1 + pad_y : max(y1 + pad_y + 1, y2 - pad_y), x1 + pad_x : max(x1 + pad_x + 1, x2 - pad_x)]
            if inner.size == 0:
                continue
            inner_wall_density = float(inner.mean())
            if inner_wall_density < 0.22:
                boxes.append({"pixels": cell_w * cell_h, "bbox": (x1, y1, x2, y2)})
    return _classify_boxes_as_rooms(boxes, base_area, width, height, "line-grid")


def _color_region_rooms(image, base_area):
    rgb = np.array(image.convert("RGB"))
    height, width, _ = rgb.shape
    max_channel = rgb.max(axis=2).astype(int)
    min_channel = rgb.min(axis=2).astype(int)
    saturation = max_channel - min_channel
    brightness = rgb.mean(axis=2)
    colored = (saturation > 28) & (brightness > 35) & (brightness < 242)
    if float(colored.mean()) < 0.04:
        return []

    quantized = (rgb // 32).astype(np.uint8)
    colors, counts = np.unique(quantized[colored].reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    boxes = []
    min_pixels = max(80, int(width * height * 0.002))
    for color in colors[order[:16]]:
        color_mask = colored & np.all(quantized == color, axis=2)
        color_mask = _binary_dilate(color_mask, iterations=1)
        components = _connected_components(color_mask, min_pixels)
        for component in components:
            min_x, min_y, max_x, max_y = component["bbox"]
            if component["pixels"] < min_pixels:
                continue
            if (max_x - min_x + 1) * (max_y - min_y + 1) > width * height * 0.55:
                continue
            boxes.append(component)
    if len(boxes) < 3:
        return []
    boxes = sorted(boxes, key=lambda item: item["pixels"], reverse=True)[:14]
    return _classify_boxes_as_rooms(boxes, base_area, width, height, "color-region")


def _rooms_score(rooms, wall_density):
    if not rooms:
        return 0
    count = len(rooms)
    count_score = 1 - min(abs(count - 6), 8) / 8
    type_score = 0
    types = {room["type"] for room in rooms}
    if "living" in types:
        type_score += 0.25
    if "kitchen" in types:
        type_score += 0.2
    if "bedroom" in types:
        type_score += 0.25
    if "bathroom" in types:
        type_score += 0.2
    wall_score = 1 - min(abs(wall_density - 0.16), 0.28) / 0.28
    over_penalty = max(0, count - 10) * 0.08
    return count_score * 0.48 + type_score * 0.34 + wall_score * 0.18 - over_penalty


def _segment_count_summary(rooms):
    if not rooms:
        return {"primary": 0, "wet": 0, "service": 0, "count_source": "none"}
    primary_types = {"bedroom", "living", "kitchen"}
    wet_types = {"bathroom"}
    service_types = {"balcony", "utility", "storage", "service"}
    primary = [room for room in rooms if room.get("type") in primary_types]
    wet = [room for room in rooms if room.get("type") in wet_types]
    service = [room for room in rooms if room.get("type") in service_types]
    areas = sorted([float(room.get("area_sqft") or 0) for room in rooms if float(room.get("area_sqft") or 0) > 0])
    median_area = areas[len(areas) // 2] if areas else 0
    plausible_primary = [
        room
        for room in primary
        if not median_area or float(room.get("area_sqft") or 0) >= median_area * 0.45
    ]
    primary_count = len(plausible_primary) or len(primary)
    if len(rooms) > 10:
        primary_count = min(primary_count, max(3, round(len(rooms) * 0.55)))
    return {
        "primary": max(0, min(primary_count, 10)),
        "wet": len(wet),
        "service": len(service),
        "count_source": "segment-calibrated",
    }


def _apply_ocr_labels_to_rooms(rooms, labels):
    if not rooms or not labels:
        return rooms
    if all(str(room.get("source", "")).startswith(("ocr-anchored", "wall-line-snap")) for room in rooms):
        return rooms
    labeled = [dict(room) for room in rooms]
    used_rooms = set()
    label_counters = {"bedroom": 0, "bathroom": 0, "kitchen": 0, "living": 0, "balcony": 0, "service": 0}

    def room_center(room):
        return room["x"] + room["width"] / 2, room["y"] + room["height"] / 2

    expected_area = {
        "bathroom": 60,
        "kitchen": 110,
        "bedroom": 130,
        "living": 240,
        "balcony": 90,
        "service": 55,
    }

    for label in sorted(labels, key=lambda item: item.get("confidence", 0), reverse=True):
        best_idx = None
        best_score = -999
        for idx, room in enumerate(labeled):
            if idx in used_rooms:
                continue
            cx = float(label.get("cx", 0))
            cy = float(label.get("cy", 0))
            inside = room["x"] <= cx <= room["x"] + room["width"] and room["y"] <= cy <= room["y"] + room["height"]
            rcx, rcy = room_center(room)
            distance = math.hypot(cx - rcx, cy - rcy)
            area = max(float(room.get("area_sqft") or 1), 1)
            target = expected_area.get(label["type"], 110)
            area_penalty = min(28, abs(area - target) / target * 12)
            if label["type"] in {"bathroom", "service"} and area > 150:
                area_penalty += 18
            if label["type"] == "kitchen" and area > 320:
                area_penalty += 18
            if label["type"] == "bedroom" and area < 45:
                area_penalty += 10
            if label["type"] == "living" and area > 160:
                area_penalty *= 0.45
            score = (22 if inside else 0) - distance * 0.65 - area_penalty
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None or best_score < -8:
            continue
        label_counters[label["type"]] = label_counters.get(label["type"], 0) + 1
        number = label_counters[label["type"]]
        if label["type"] == "bedroom":
            room_label = f"Bedroom {number}"
        elif label["type"] == "bathroom":
            room_label = f"Bathroom {number}"
        elif label["type"] == "living":
            room_label = "Living / Hall" if number == 1 else f"Common Zone {number}"
        elif label["type"] == "balcony":
            room_label = "Balcony / Veranda" if number == 1 else f"Balcony / Veranda {number}"
        elif label["type"] == "service":
            room_label = "Service Zone" if number == 1 else f"Service Zone {number}"
        else:
            room_label = label["label"]
        labeled[best_idx].update({
            "type": label["type"],
            "label": room_label,
            "confidence": max(float(labeled[best_idx].get("confidence", 0.55)), min(0.88, float(label.get("confidence", 45)) / 100 + 0.18)),
            "ocr_label_text": label.get("text", ""),
            "label_x": round(float(label.get("cx") or 0), 1),
            "label_y": round(float(label.get("cy") or 0), 1),
            "source": f"{labeled[best_idx].get('source', 'geometry')}+ocr",
        })
        used_rooms.add(best_idx)
    return labeled


def _rooms_from_ocr_labels(labels, base_area, label_hints):
    if not labels:
        return []
    caps = {
        "bedroom": max(0, int(label_hints.get("bedrooms") or 0)),
        "bathroom": max(0, int(label_hints.get("bathrooms") or 0)),
        "kitchen": max(0, int(label_hints.get("kitchens") or 0)),
        "living": max(0, int(label_hints.get("halls") or 0)),
        "balcony": max(0, int(label_hints.get("outdoor_zones") or 0)),
        "service": max(0, int(label_hints.get("service_zones") or 0) + int(label_hints.get("stair_zones") or 0)),
    }
    defaults = {
        "bedroom": (18, 14, 120),
        "bathroom": (10, 8, 55),
        "kitchen": (14, 11, 95),
        "living": (30, 18, 220),
        "balcony": (22, 10, 90),
        "service": (11, 8, 45),
    }
    counters = {key: 0 for key in caps}
    rooms = []
    for label in sorted(labels, key=lambda item: item.get("confidence", 0), reverse=True):
        room_type = label.get("type")
        if room_type not in defaults:
            continue
        if caps.get(room_type, 0) and counters[room_type] >= caps[room_type]:
            continue
        if not caps.get(room_type, 0) and room_type in {"bedroom", "bathroom", "kitchen"}:
            continue
        width, height, area = defaults[room_type]
        cx = float(label.get("cx") or 62.5)
        cy = float(label.get("cy") or 46)
        x = max(3, min(125 - width - 3, cx - width / 2))
        y = max(3, min(92 - height - 3, cy - height / 2))
        counters[room_type] += 1
        count = counters[room_type]
        if room_type == "bedroom":
            room_label = f"Bedroom {count}"
        elif room_type == "bathroom":
            room_label = f"Bathroom {count}"
        elif room_type == "living":
            room_label = "Living / Hall" if count == 1 else f"Common Zone {count}"
        elif room_type == "balcony":
            room_label = "Balcony / Veranda" if count == 1 else f"Balcony / Veranda {count}"
        elif room_type == "service":
            room_label = "Service Zone" if count == 1 else f"Service Zone {count}"
        else:
            room_label = label.get("label") or "Kitchen"
        rooms.append({
            "id": f"ocr-zone-{len(rooms) + 1}",
            "type": room_type,
            "label": room_label,
            "x": round(x, 1),
            "y": round(y, 1),
            "width": round(width, 1),
            "height": round(height, 1),
            "area_sqft": round(area * math.sqrt(max(base_area, 1) / 1250), 1),
            "confidence": round(min(0.9, max(0.62, float(label.get("confidence", 55)) / 100)), 2),
            "polygon": [[round(x, 1), round(y, 1)], [round(x + width, 1), round(y, 1)], [round(x + width, 1), round(y + height, 1)], [round(x, 1), round(y + height, 1)]],
            "label_x": round(cx, 1),
            "label_y": round(cy, 1),
            "source": "ocr-spatial-label",
            "ocr_label_text": label.get("text", ""),
        })
    total_area = sum(float(room["area_sqft"]) for room in rooms)
    target_area = max(1, float(base_area) * 0.82)
    if total_area:
        scale = target_area / total_area
        for room in rooms:
            room["area_sqft"] = round(float(room["area_sqft"]) * scale, 1)
    return rooms


def _augment_rooms_from_hints(rooms, base_area, label_hints):
    if not rooms or not label_hints or not label_hints.get("has_label_hint"):
        return rooms
    target_by_type = {
        "bedroom": int(label_hints.get("bedrooms") or 0),
        "bathroom": int(label_hints.get("bathrooms") or 0),
        "kitchen": int(label_hints.get("kitchens") or 0),
        # Do not invent living/balcony/service zones. OCR duplicates those labels
        # often, and invented service boxes made the overlay look much worse.
        "living": 0,
        "balcony": 0,
        "service": 0,
    }
    defaults = {
        "bedroom": (18, 14, 120),
        "bathroom": (9, 8, 55),
        "kitchen": (13, 11, 90),
        "living": (24, 16, 180),
        "balcony": (18, 9, 70),
        "service": (10, 8, 45),
    }
    seed_positions = [(9, 8), (35, 8), (62, 8), (89, 8), (9, 32), (35, 32), (62, 32), (89, 32), (9, 58), (35, 58), (62, 58), (89, 58)]
    augmented = [dict(room) for room in rooms]

    def overlaps(candidate):
        cx, cy, cw, ch = candidate
        for room in augmented:
            rx, ry = float(room["x"]), float(room["y"])
            rw, rh = float(room["width"]), float(room["height"])
            ix = max(0, min(cx + cw, rx + rw) - max(cx, rx))
            iy = max(0, min(cy + ch, ry + rh) - max(cy, ry))
            if ix * iy > min(cw * ch, rw * rh) * 0.35:
                return True
        return False

    for room_type, target in target_by_type.items():
        current = sum(1 for room in augmented if room.get("type") == room_type)
        missing = max(0, min(target - current, 4))
        for _ in range(missing):
            width, height, area = defaults[room_type]
            position = None
            for x, y in seed_positions:
                if not overlaps((x, y, width, height)):
                    position = (x, y)
                    break
            if position is None:
                position = (max(4, 120 - width - len(augmented) * 3), max(4, 86 - height - len(augmented) * 2))
            x, y = position
            count = sum(1 for room in augmented if room.get("type") == room_type) + 1
            augmented.append({
                "id": f"ocr-inferred-{room_type}-{count}",
                "type": room_type,
                "label": _room_label_for_type(room_type, count),
                "x": round(x, 1),
                "y": round(y, 1),
                "width": round(width, 1),
                "height": round(height, 1),
                "area_sqft": round(area * math.sqrt(max(float(base_area), 1) / 1250), 1),
                "confidence": 0.56,
                "polygon": [[round(x, 1), round(y, 1)], [round(x + width, 1), round(y, 1)], [round(x + width, 1), round(y + height, 1)], [round(x, 1), round(y + height, 1)]],
                "source": "ocr-count-inferred",
                "ocr_label_text": "Inferred from OCR count",
            })
    return augmented


def _deduped_label_count(normalized: str, patterns, max_count: int, divisor: float = 2.0):
    raw_count = sum(len(re.findall(pattern, normalized)) for pattern in patterns)
    if raw_count <= 0:
        return 0
    return max(1, min(max_count, int(round(raw_count / divisor))))


def _bedroom_label_count(normalized: str, max_count: int = 8):
    master_count = 1 if re.search(r"\bMASTER\s+BED\s+ROOM\b", normalized) else 0
    without_master = re.sub(r"\bMASTER\s+BED\s+ROOM\b", " ", normalized)
    if master_count:
        regular_count = sum(
            1
            for pattern in (r"\bBEDROOM\b", r"\bBED\s+ROOM\b", r"\bBEDRM\b")
            if re.search(pattern, without_master)
        )
    else:
        raw_regular = len(re.findall(r"\bBEDROOM\b", without_master))
        raw_regular += len(re.findall(r"\bBED\s+ROOM\b", without_master))
        raw_regular += len(re.findall(r"\bBEDRM\b", without_master))
        regular_count = max(0, min(max_count, int(round(raw_regular / 1.8)))) if raw_regular else 0
    return min(max_count, master_count + regular_count)


def _bathroom_label_count(normalized: str, max_count: int = 6):
    variants = (
        r"\bBATH\s*ROOM\b",
        r"\bBATHROOM\b",
        r"\bTOILET\b",
        r"\bTOKET\b",
        r"\bTORET\b",
        r"\bT0ILET\b",
        r"\bWC\b",
    )
    variant_count = sum(1 for pattern in variants if re.search(pattern, normalized))
    if variant_count > 1:
        return min(max_count, variant_count)
    return _deduped_label_count(normalized, variants, max_count, divisor=1.8)


def _label_hints_from_text(text):
    normalized = re.sub(r"[^A-Z0-9]+", " ", _normalize_arch_text(text))
    normalized = re.sub(r"\b(?:SIMPLE|HOUSE|PLAN|DWG|DWN|PINCADD|COM|MAIN|GATE)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    bhk_match = re.search(r"([1-6])\s*BHK", normalized)
    bedroom_numbers = set(re.findall(r"\b(?:BED(?:ROOM|RM)?|BED\s+ROOM|MASTER\s+BED\s+ROOM)\s*([1-9])\b", normalized))
    bedroom_phrase_count = _bedroom_label_count(normalized)
    if bhk_match:
        bedrooms = int(bhk_match.group(1))
    elif len(bedroom_numbers) >= 2:
        bedrooms = len(bedroom_numbers)
    else:
        bedrooms = max(len(bedroom_numbers), bedroom_phrase_count)

    bath_numbers = set(re.findall(r"\b(?:BATH(?:ROOM)?|TOILET|TOKET|TORET|T0ILET|WC)\s*([1-9])\b", normalized))
    bathroom_phrase_count = _bathroom_label_count(normalized)
    bathrooms = len(bath_numbers) if len(bath_numbers) >= 2 else max(len(bath_numbers), bathroom_phrase_count)

    kitchens = 1 if re.search(r"\b(?:KITCHEN|KIT)\b", normalized) else 0
    common_zone_patterns = ("LIVING", "HALL", "HALLWAY", "LOUNGE", "DINING", "SITTING", "FAMILY")
    halls = sum(1 for pattern in common_zone_patterns if re.search(rf"\b{pattern}\b", normalized))
    outdoor_patterns = ("VERANDA", "BALCONY", "PORCH", "TERRACE")
    outdoor_zones = sum(1 for pattern in outdoor_patterns if re.search(rf"\b{pattern}\b", normalized))
    service_patterns = ("STORE", "STORAGE", "UTILITY", "UTILITIES", "WASH", "LAUNDRY")
    service_zones = sum(1 for pattern in service_patterns if re.search(rf"\b{pattern}\b", normalized))
    stair_zones = 1 if re.search(r"\b(?:STAIR(?:CASE)?|STAR)\b", normalized) else 0

    primary_room_count = bedrooms + kitchens + halls + outdoor_zones
    named_zone_count = primary_room_count + bathrooms + service_zones + stair_zones
    area_matches = [float(match) for match in re.findall(r"(\d{3,6}(?:\.\d+)?)\s*(?:SQFT|SQ FT|SQ\.FT|SOFT)", normalized)]
    if len(area_matches) >= 3:
        capped = []
        for value in area_matches:
            if all(abs(value - existing) > 1.5 for existing in capped):
                capped.append(value)
        total_area = sum(capped)
    else:
        total_area = max(area_matches) if area_matches else None
    return {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "kitchens": kitchens,
        "halls": halls,
        "outdoor_zones": outdoor_zones,
        "service_zones": service_zones,
        "stair_zones": stair_zones,
        "primary_room_count": primary_room_count,
        "named_zone_count": named_zone_count,
        "total_area": total_area,
        "has_label_hint": any([bedrooms, bathrooms, kitchens, halls, outdoor_zones, service_zones, stair_zones, area_matches]),
    }


def _merge_label_hints(*hints):
    merged = {
        "bedrooms": 0,
        "bathrooms": 0,
        "kitchens": 0,
        "halls": 0,
        "outdoor_zones": 0,
        "service_zones": 0,
        "stair_zones": 0,
        "primary_room_count": 0,
        "named_zone_count": 0,
        "total_area": None,
        "has_label_hint": False,
    }
    for hint in hints:
        if not hint:
            continue
        for key in ("bedrooms", "bathrooms", "kitchens", "halls", "outdoor_zones", "service_zones", "stair_zones", "primary_room_count", "named_zone_count"):
            merged[key] = max(int(merged.get(key) or 0), int(hint.get(key) or 0))
        if hint.get("total_area"):
            merged["total_area"] = hint["total_area"] if not merged["total_area"] else max(merged["total_area"], hint["total_area"])
        merged["has_label_hint"] = merged["has_label_hint"] or bool(hint.get("has_label_hint"))
    merged["primary_room_count"] = merged["bedrooms"] + merged["kitchens"] + merged["halls"] + merged["outdoor_zones"]
    merged["named_zone_count"] = merged["primary_room_count"] + merged["bathrooms"] + merged["service_zones"] + merged["stair_zones"]
    merged["has_label_hint"] = merged["has_label_hint"] or merged["named_zone_count"] > 0
    return merged


def _choose_plan_area(dxf: dict, label_hints: dict, dimension_evidence: dict, project_area: float | None):
    dxf = dxf or {}
    label_hints = label_hints or {}
    dimension_evidence = dimension_evidence or {}
    dimension_edge_area = _edge_area_from_dimension_evidence(dimension_evidence) if int(dimension_evidence.get("edge_binding_axes") or 0) >= 2 else None
    named_zone_count = max(
        int(label_hints.get("named_zone_count") or 0),
        int(dxf.get("bedrooms") or 0) + int(dxf.get("bathrooms") or 0) + int(dxf.get("kitchens") or 0) + int(dxf.get("halls") or 0),
    )
    largest_room_area = max(
        float(dxf.get("largest_room_area_sqft") or 0),
        max((float(value) for value in (dimension_evidence.get("room_area_values_sqft") or [])), default=0.0),
    )
    minimum_reasonable_total = 0.0
    if named_zone_count >= 3:
        minimum_reasonable_total = max(350.0, largest_room_area * 1.7 if largest_room_area else 0.0)

    def trustworthy(candidate):
        if not candidate:
            return False
        area = float(candidate)
        if minimum_reasonable_total and area < minimum_reasonable_total:
            return False
        if project_area and named_zone_count >= 3 and area < float(project_area) * 0.6:
            return False
        return True

    if trustworthy(dxf.get("total_area")):
        return float(dxf["total_area"])
    if trustworthy(label_hints.get("total_area")):
        return float(label_hints["total_area"])
    if project_area:
        return float(project_area)
    if trustworthy(dimension_edge_area):
        return float(dimension_edge_area)
    if dxf.get("room_area_sum_sqft") and trustworthy(dxf["room_area_sum_sqft"]):
        return float(dxf["room_area_sum_sqft"])
    if dxf.get("total_area"):
        return float(dxf["total_area"])
    if label_hints.get("total_area"):
        return float(label_hints["total_area"])
    return 1250.0


def _dimension_token_to_ft(token: str):
    text = _normalize_arch_text(token).strip()
    text = re.sub(r"\s+", "", text)
    match = re.match(r"^(\d{1,3})(?:[-']+(\d{1,2}))?(?:\"|''|IN)?$", text)
    if match and ("'" in text or '"' in text or "-" in text):
        feet = float(match.group(1))
        inches = float(match.group(2) or 0)
        return feet + inches / 12
    match = re.match(r"^(\d{1,3})\s*FT$", text)
    if match:
        return float(match.group(1))
    return None


def _dedupe_dimension_boxes(dimension_boxes):
    deduped = []
    for box in sorted(dimension_boxes or [], key=lambda item: item.get("confidence", 0), reverse=True):
        duplicate = False
        for existing in deduped:
            close = math.hypot(float(existing["cx"]) - float(box["cx"]), float(existing["cy"]) - float(box["cy"])) < 4.5
            same_value = abs(float(existing["value_ft"]) - float(box["value_ft"])) < 0.35
            if close and same_value:
                duplicate = True
                break
        if not duplicate:
            deduped.append(box)
    return deduped[:36]


def _edge_dimension_sum(boxes, edge):
    candidates = [box for box in boxes if box.get("edge") == edge]
    candidates.sort(key=lambda item: item["cx"] if edge in {"top", "bottom"} else item["cy"])
    unique = []
    for box in candidates:
        axis = float(box["cx"] if edge in {"top", "bottom"} else box["cy"])
        value = float(box["value_ft"])
        if all(abs(axis - seen["axis"]) > 3.0 or abs(value - seen["value"]) > 0.35 for seen in unique):
            unique.append({"axis": axis, "value": value})
    if len(unique) >= 2:
        return round(sum(item["value"] for item in unique), 2), len(unique)
    if len(unique) == 1:
        return round(unique[0]["value"], 2), 1
    return None, 0


def _edge_bound_span(edge_sums: dict | None, edges: tuple[str, ...]):
    values = [float((edge_sums or {}).get(edge) or 0) for edge in edges if (edge_sums or {}).get(edge)]
    return round(max(values), 2) if values else None


def _edge_area_from_dimension_evidence(dimension_evidence: dict | None):
    edge_sums = (dimension_evidence or {}).get("edge_sums_ft") or {}
    width = (dimension_evidence or {}).get("exterior_width_ft") or _edge_bound_span(edge_sums, ("top", "bottom"))
    depth = (dimension_evidence or {}).get("exterior_depth_ft") or _edge_bound_span(edge_sums, ("left", "right"))
    if width and depth:
        return round(float(width) * float(depth), 1)
    return None


def _extract_dimension_evidence(text: str, base_area: float | None = None, dimension_boxes=None):
    normalized = _normalize_arch_text(text)
    raw_tokens = re.findall(r"\b\d{1,3}\s*(?:'|-)\s*\d{1,2}\s*(?:\"|IN)?|\b\d{1,3}\s*'\s*|\b\d{1,3}\s*FT\b", normalized)
    dimensions = []
    for token in raw_tokens:
        value = _dimension_token_to_ft(token)
        if value and 2 <= value <= 80:
            dimensions.append(round(value, 2))
    area_values = [float(match) for match in re.findall(r"(\d{2,6}(?:\.\d+)?)\s*(?:SQ\s*FT|SQFT|SQ\.FT|SOFT)", normalized)]
    room_area_values = [value for value in area_values if 20 <= value <= 1200]
    total_area = max(room_area_values) if room_area_values else None
    if len(room_area_values) >= 3:
        # Repeated OCR passes duplicate the same room areas; unique room-area values
        # are a better demo proxy than summing every OCR occurrence.
        unique_areas = []
        for value in room_area_values:
            if all(abs(value - existing) > 1.5 for existing in unique_areas):
                unique_areas.append(value)
        if unique_areas:
            total_area = sum(unique_areas)
            if base_area and total_area > float(base_area) * 1.35:
                total_area = float(base_area)
    unique_dims = []
    for value in dimensions:
        if all(abs(value - existing) > 0.35 for existing in unique_dims):
            unique_dims.append(value)
    boxes = _dedupe_dimension_boxes(dimension_boxes or [])
    edge_sums = {}
    edge_counts = {}
    for edge in ("top", "bottom", "left", "right"):
        edge_value, edge_count = _edge_dimension_sum(boxes, edge)
        if edge_value:
            edge_sums[edge] = edge_value
            edge_counts[edge] = edge_count

    horizontal_values = [edge_sums[edge] for edge in ("top", "bottom") if edge in edge_sums]
    vertical_values = [edge_sums[edge] for edge in ("left", "right") if edge in edge_sums]
    edge_area = round(max(horizontal_values) * max(vertical_values), 1) if horizontal_values and vertical_values else None
    exterior_width = max(horizontal_values) if horizontal_values else None
    exterior_depth = max(vertical_values) if vertical_values else None
    sorted_dims = sorted(unique_dims, reverse=True)
    if not exterior_width and sorted_dims and len(unique_dims) >= 2:
        exterior_width = sorted_dims[0]
    if not exterior_depth and len(sorted_dims) > 1:
        exterior_depth = sorted_dims[1]
    reference_area = float(base_area or total_area or 0)
    if exterior_width and exterior_depth and reference_area and exterior_width * exterior_depth < 0.35 * reference_area:
        if horizontal_values and not vertical_values:
            exterior_depth = None
        elif vertical_values and not horizontal_values:
            exterior_width = None
        else:
            exterior_width = exterior_depth = None
    confidence = 0
    if len(unique_dims) >= 2:
        confidence += 35
    if len(unique_dims) >= 5:
        confidence += 20
    if total_area:
        confidence += 25
    if edge_sums:
        confidence += 20
    if exterior_width and exterior_depth:
        confidence += 20
    if horizontal_values and vertical_values and exterior_width and exterior_depth:
        confidence = max(confidence, 82)
    elif horizontal_values and vertical_values:
        confidence = max(confidence, 74)
    return {
        "dimensions_ft": sorted(unique_dims),
        "dimension_count": len(unique_dims),
        "dimension_boxes": boxes,
        "edge_sums_ft": edge_sums,
        "edge_dimension_counts": edge_counts,
        "edge_binding_axes": int(bool(horizontal_values)) + int(bool(vertical_values)),
        "room_area_values_sqft": room_area_values[:24],
        "area_from_text_sqft": round(total_area, 1) if total_area else None,
        "edge_area_sqft": edge_area,
        "exterior_width_ft": round(exterior_width, 2) if exterior_width else None,
        "exterior_depth_ft": round(exterior_depth, 2) if exterior_depth else None,
        "confidence": min(100, confidence),
    }


def _dxf_dimension_evidence(dxf_meta: dict | None, base_area: float | None = None):
    dxf_meta = dxf_meta or {}
    evidence = dict(dxf_meta.get("text_dimension_evidence") or {})
    evidence["source_prefix"] = "dxf"
    geometry_bounds = dxf_meta.get("geometry_bounds") or {}
    width_units = float(geometry_bounds.get("width_units") or 0)
    height_units = float(geometry_bounds.get("height_units") or 0)
    if width_units and height_units:
        aspect_hint = max(0.55, min(2.4, width_units / height_units))
        evidence["aspect_hint"] = round(aspect_hint, 3)
        reference_area = float(base_area or dxf_meta.get("total_area") or 0)
        if reference_area and not (evidence.get("exterior_width_ft") and evidence.get("exterior_depth_ft")):
            width_ft = math.sqrt(max(reference_area, 1) * aspect_hint)
            depth_ft = reference_area / max(width_ft, 0.1)
            evidence["exterior_width_ft"] = round(width_ft, 2)
            evidence["exterior_depth_ft"] = round(depth_ft, 2)
            evidence["calibrated_from_area"] = True
        geometry_confidence = 34 + min(int(dxf_meta.get("line_count") or 0) // 4, 18)
        if base_area or dxf_meta.get("total_area"):
            geometry_confidence += 10
        evidence["confidence"] = max(float(evidence.get("confidence") or 0), min(78, geometry_confidence))
    return evidence


def _merge_dimension_evidence(primary: dict | None, fallback: dict | None):
    primary = dict(primary or {})
    fallback = dict(fallback or {})
    merged = dict(primary)
    for key in ("exterior_width_ft", "exterior_depth_ft", "aspect_hint", "calibrated_from_area", "source_prefix"):
        if merged.get(key) in (None, "", 0):
            fallback_value = fallback.get(key)
            if fallback_value not in (None, "", 0):
                merged[key] = fallback_value
    merged["dimensions_ft"] = sorted(
        {
            round(float(value), 2)
            for value in [*(primary.get("dimensions_ft") or []), *(fallback.get("dimensions_ft") or [])]
        }
    )
    merged["dimension_count"] = len(merged["dimensions_ft"])
    merged["confidence"] = round(max(float(primary.get("confidence") or 0), float(fallback.get("confidence") or 0)), 1)
    merged["dimension_boxes"] = _dedupe_dimension_boxes([*(primary.get("dimension_boxes") or []), *(fallback.get("dimension_boxes") or [])])
    merged["edge_sums_ft"] = {
        **(fallback.get("edge_sums_ft") or {}),
        **(primary.get("edge_sums_ft") or {}),
    }
    merged["edge_dimension_counts"] = {
        edge: max(
            int((primary.get("edge_dimension_counts") or {}).get(edge) or 0),
            int((fallback.get("edge_dimension_counts") or {}).get(edge) or 0),
        )
        for edge in set((primary.get("edge_dimension_counts") or {}).keys()) | set((fallback.get("edge_dimension_counts") or {}).keys())
    }
    merged["edge_binding_axes"] = max(
        int(primary.get("edge_binding_axes") or 0),
        int(fallback.get("edge_binding_axes") or 0),
        int(bool(_edge_bound_span(merged["edge_sums_ft"], ("top", "bottom")))) + int(bool(_edge_bound_span(merged["edge_sums_ft"], ("left", "right")))),
    )
    merged["room_area_values_sqft"] = primary.get("room_area_values_sqft") or fallback.get("room_area_values_sqft") or []
    merged["area_from_text_sqft"] = primary.get("area_from_text_sqft") or fallback.get("area_from_text_sqft")
    merged["edge_area_sqft"] = primary.get("edge_area_sqft") or fallback.get("edge_area_sqft") or _edge_area_from_dimension_evidence(merged)
    if primary.get("exterior_width_ft") or primary.get("exterior_depth_ft"):
        merged["source_prefix"] = primary.get("source_prefix") or "ocr"
    elif fallback:
        merged["source_prefix"] = fallback.get("source_prefix") or "dxf"
    return merged


def _polygon_perimeter_units(room: dict):
    polygon = room.get("polygon") or []
    if len(polygon) < 3:
        return 2 * (float(room.get("width") or 0) + float(room.get("height") or 0))
    perimeter = 0
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        perimeter += math.hypot(float(next_point[0]) - float(point[0]), float(next_point[1]) - float(point[1]))
    return perimeter


def _room_bounds(room: dict):
    polygon = room.get("polygon") or []
    if len(polygon) >= 3:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
    else:
        min_x = float(room.get("x") or 0)
        min_y = float(room.get("y") or 0)
        max_x = min_x + float(room.get("width") or 0)
        max_y = min_y + float(room.get("height") or 0)
    return {
        "min_x": round(min_x, 2),
        "max_x": round(max_x, 2),
        "min_y": round(min_y, 2),
        "max_y": round(max_y, 2),
        "width": round(max(0.01, max_x - min_x), 2),
        "height": round(max(0.01, max_y - min_y), 2),
    }


def _bbox_polygon(bounds: dict):
    return [
        [round(bounds["min_x"], 1), round(bounds["min_y"], 1)],
        [round(bounds["max_x"], 1), round(bounds["min_y"], 1)],
        [round(bounds["max_x"], 1), round(bounds["max_y"], 1)],
        [round(bounds["min_x"], 1), round(bounds["max_y"], 1)],
    ]


def _merge_axis_anchors(values, tolerance=2.4):
    values = sorted(float(value) for value in values)
    if not values:
        return []
    groups = [[values[0]]]
    for value in values[1:]:
        if abs(value - groups[-1][-1]) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(sum(group) / len(group), 2) for group in groups]


def _snap_to_anchor(value, anchors, tolerance=2.8):
    if not anchors:
        return float(value)
    best = min(anchors, key=lambda anchor: abs(float(value) - anchor))
    return best if abs(float(value) - best) <= tolerance else float(value)


def _room_overlap_units(bounds_a: dict, bounds_b: dict):
    overlap_w = max(0.0, min(bounds_a["max_x"], bounds_b["max_x"]) - max(bounds_a["min_x"], bounds_b["min_x"]))
    overlap_h = max(0.0, min(bounds_a["max_y"], bounds_b["max_y"]) - max(bounds_a["min_y"], bounds_b["min_y"]))
    return overlap_w, overlap_h, overlap_w * overlap_h


def _shared_wall_units(bounds_a: dict, bounds_b: dict, tolerance=2.2):
    vertical_candidates = []
    if abs(bounds_a["max_x"] - bounds_b["min_x"]) <= tolerance:
        vertical_candidates.append(max(0.0, min(bounds_a["max_y"], bounds_b["max_y"]) - max(bounds_a["min_y"], bounds_b["min_y"])))
    if abs(bounds_b["max_x"] - bounds_a["min_x"]) <= tolerance:
        vertical_candidates.append(max(0.0, min(bounds_a["max_y"], bounds_b["max_y"]) - max(bounds_a["min_y"], bounds_b["min_y"])))
    horizontal_candidates = []
    if abs(bounds_a["max_y"] - bounds_b["min_y"]) <= tolerance:
        horizontal_candidates.append(max(0.0, min(bounds_a["max_x"], bounds_b["max_x"]) - max(bounds_a["min_x"], bounds_b["min_x"])))
    if abs(bounds_b["max_y"] - bounds_a["min_y"]) <= tolerance:
        horizontal_candidates.append(max(0.0, min(bounds_a["max_x"], bounds_b["max_x"]) - max(bounds_a["min_x"], bounds_b["min_x"])))
    return max(vertical_candidates + horizontal_candidates + [0.0])


def _room_label_anchor(room: dict, bounds: dict | None = None):
    bounds = bounds or _room_bounds(room)
    explicit_x = room.get("label_x")
    explicit_y = room.get("label_y")
    if explicit_x is not None and explicit_y is not None:
        x = float(explicit_x)
        y = float(explicit_y)
        if bounds["min_x"] + 1 <= x <= bounds["max_x"] - 1 and bounds["min_y"] + 2 <= y <= bounds["max_y"] - 2:
            return round(x, 1), round(y, 1)
    x = max(bounds["min_x"] + 1.8, min(bounds["max_x"] - 10.0, bounds["min_x"] + min(bounds["width"] * 0.18, 7.0)))
    y = max(bounds["min_y"] + 5.2, min(bounds["max_y"] - 4.8, bounds["min_y"] + min(bounds["height"] * 0.24, 7.4)))
    if bounds["width"] < 12:
        x = bounds["min_x"] + 1.3
    return round(x, 1), round(y, 1)


def _anchor_interval_around_value(value: float, minimum: float, maximum: float, anchors: list[float]):
    points = sorted(set([minimum, maximum, *[float(anchor) for anchor in anchors if minimum < float(anchor) < maximum]]))
    if len(points) < 2:
        return minimum, maximum
    for index in range(len(points) - 1):
        left = points[index]
        right = points[index + 1]
        if left <= value <= right:
            return left, right
    return minimum, maximum


def _tighten_room_bounds_to_label_cell(room: dict, bounds: dict, x_anchors: list[float], y_anchors: list[float]):
    bounds = {
        **bounds,
        "width": round(max(0.01, float(bounds["max_x"]) - float(bounds["min_x"])), 2),
        "height": round(max(0.01, float(bounds["max_y"]) - float(bounds["min_y"])), 2),
    }
    source = str(room.get("source") or "")
    if not source.startswith(("ocr-anchor", "ocr-anchored", "wall-line-snap", "label-fallback-fit")):
        return bounds, False
    internal_x = [value for value in x_anchors if bounds["min_x"] + 1.6 < value < bounds["max_x"] - 1.6]
    internal_y = [value for value in y_anchors if bounds["min_y"] + 1.6 < value < bounds["max_y"] - 1.6]
    if not internal_x and not internal_y:
        return bounds, False
    room_label = str(room.get("label") or "")
    generic_zone = room_label.startswith("Common Zone") or room_label.startswith("Service Zone")
    confidence = float(room.get("confidence") or 0)
    if not generic_zone and confidence >= 0.76 and len(internal_x) + len(internal_y) < 3:
        return bounds, False

    label_x, label_y = _room_label_anchor(room, bounds)
    tightened = dict(bounds)
    if internal_x:
        left, right = _anchor_interval_around_value(label_x, bounds["min_x"], bounds["max_x"], internal_x)
        max_width = bounds["width"] * (0.9 if generic_zone else 0.78)
        if right - left >= 4.0 and right - left <= max_width:
            tightened["min_x"] = left
            tightened["max_x"] = right
    if internal_y:
        top, bottom = _anchor_interval_around_value(label_y, bounds["min_y"], bounds["max_y"], internal_y)
        max_height = bounds["height"] * (0.9 if generic_zone else 0.8)
        if bottom - top >= 4.0 and bottom - top <= max_height:
            tightened["min_y"] = top
            tightened["max_y"] = bottom

    tightened["min_x"] = max(0.0, min(121.0, tightened["min_x"]))
    tightened["max_x"] = max(tightened["min_x"] + 4.0, min(125.0, tightened["max_x"]))
    tightened["min_y"] = max(0.0, min(88.0, tightened["min_y"]))
    tightened["max_y"] = max(tightened["min_y"] + 4.0, min(92.0, tightened["max_y"]))
    tightened["width"] = round(max(0.01, tightened["max_x"] - tightened["min_x"]), 2)
    tightened["height"] = round(max(0.01, tightened["max_y"] - tightened["min_y"]), 2)
    changed = (
        abs(tightened["min_x"] - bounds["min_x"]) > 0.2
        or abs(tightened["max_x"] - bounds["max_x"]) > 0.2
        or abs(tightened["min_y"] - bounds["min_y"]) > 0.2
        or abs(tightened["max_y"] - bounds["max_y"]) > 0.2
    )
    return (tightened if changed else bounds), changed


def _label_anchor_candidates(room: dict, bounds: dict):
    explicit = room.get("label_x"), room.get("label_y")
    candidates = []
    if explicit[0] is not None and explicit[1] is not None:
        x = float(explicit[0])
        y = float(explicit[1])
        if bounds["min_x"] + 1 <= x <= bounds["max_x"] - 1 and bounds["min_y"] + 2 <= y <= bounds["max_y"] - 2:
            candidates.append((round(x, 1), round(y, 1)))
    left = round(bounds["min_x"] + min(max(bounds["width"] * 0.14, 1.4), 6.8), 1)
    center = round((bounds["min_x"] + bounds["max_x"]) / 2, 1)
    right = round(bounds["max_x"] - min(max(bounds["width"] * 0.18, 2.2), 8.6), 1)
    top = round(min(bounds["max_y"] - 5.0, bounds["min_y"] + min(max(bounds["height"] * 0.24, 5.0), 7.8)), 1)
    middle = round((bounds["min_y"] + bounds["max_y"]) / 2, 1)
    for candidate in ((left, top), (center, top), (right, top), (left, middle), (center, middle)):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _deconflict_label_anchors(rooms: list):
    if not rooms:
        return 0
    chosen = []
    for index in sorted(
        range(len(rooms)),
        key=lambda idx: (
            float(rooms[idx].get("graph_degree") or 0),
            float(rooms[idx].get("confidence") or 0),
            float(rooms[idx].get("area_sqft") or 0),
        ),
        reverse=True,
    ):
        room = rooms[index]
        bounds = room.get("bbox") or _room_bounds(room)
        center_x = (bounds["min_x"] + bounds["max_x"]) / 2
        center_y = (bounds["min_y"] + bounds["max_y"]) / 2
        best = None
        best_score = None
        for candidate in _label_anchor_candidates(room, bounds):
            min_distance = min((math.hypot(candidate[0] - point[0], candidate[1] - point[1]) for point in chosen), default=999.0)
            score = min_distance - math.hypot(candidate[0] - center_x, candidate[1] - center_y) * 0.08
            if best is None or score > best_score:
                best = candidate
                best_score = score
        if best:
            room["label_x"] = round(best[0], 1)
            room["label_y"] = round(best[1], 1)
            room["label_area_y"] = round(min(bounds["max_y"] - 1.4, best[1] + 4.3), 1)
            chosen.append(best)
    conflicts = 0
    for left in range(len(rooms)):
        left_room = rooms[left]
        left_anchor = (float(left_room.get("label_x") or 0), float(left_room.get("label_y") or 0))
        for right in range(left + 1, len(rooms)):
            right_room = rooms[right]
            right_anchor = (float(right_room.get("label_x") or 0), float(right_room.get("label_y") or 0))
            if math.hypot(left_anchor[0] - right_anchor[0], left_anchor[1] - right_anchor[1]) < 5.4:
                conflicts += 1
    return conflicts


def _refine_rooms_with_wall_graph(rooms: list):
    if not rooms:
        return {
            "rooms": [],
            "graph": {
                "edge_count": 0,
                "shared_wall_units": 0.0,
                "overlap_area_units": 0.0,
                "overlap_ratio": 0.0,
                "connectivity_score": 0.0,
                "envelope_width_units": 0.0,
                "envelope_height_units": 0.0,
                "tightened_room_count": 0,
                "label_conflict_count": 0,
            },
        }

    refined = [dict(room) for room in rooms]
    x_anchors = _merge_axis_anchors([value for room in refined for value in (_room_bounds(room)["min_x"], _room_bounds(room)["max_x"])])
    y_anchors = _merge_axis_anchors([value for room in refined for value in (_room_bounds(room)["min_y"], _room_bounds(room)["max_y"])])
    tightened_room_count = 0

    for room in refined:
        bounds = _room_bounds(room)
        snapped = {
            "min_x": max(0.0, min(121.0, _snap_to_anchor(bounds["min_x"], x_anchors))),
            "max_x": max(4.0, min(125.0, _snap_to_anchor(bounds["max_x"], x_anchors))),
            "min_y": max(0.0, min(88.0, _snap_to_anchor(bounds["min_y"], y_anchors))),
            "max_y": max(4.0, min(92.0, _snap_to_anchor(bounds["max_y"], y_anchors))),
        }
        if snapped["max_x"] - snapped["min_x"] < 4:
            snapped["max_x"] = min(125.0, snapped["min_x"] + 4.0)
        if snapped["max_y"] - snapped["min_y"] < 4:
            snapped["max_y"] = min(92.0, snapped["min_y"] + 4.0)
        snapped, changed = _tighten_room_bounds_to_label_cell(room, snapped, x_anchors, y_anchors)
        if changed:
            tightened_room_count += 1
        room["x"] = round(snapped["min_x"], 1)
        room["y"] = round(snapped["min_y"], 1)
        room["width"] = round(snapped["max_x"] - snapped["min_x"], 1)
        room["height"] = round(snapped["max_y"] - snapped["min_y"], 1)
        room["polygon"] = _bbox_polygon({
            "min_x": snapped["min_x"],
            "max_x": snapped["max_x"],
            "min_y": snapped["min_y"],
            "max_y": snapped["max_y"],
        })

    by_confidence = sorted(range(len(refined)), key=lambda idx: float(refined[idx].get("confidence") or 0), reverse=True)
    for anchor_position, anchor_idx in enumerate(by_confidence):
        for target_idx in by_confidence[anchor_position + 1 :]:
            anchor_bounds = _room_bounds(refined[anchor_idx])
            target_bounds = _room_bounds(refined[target_idx])
            overlap_w, overlap_h, overlap_area = _room_overlap_units(anchor_bounds, target_bounds)
            target_area = max(target_bounds["width"] * target_bounds["height"], 1.0)
            if overlap_area <= 0 or overlap_area / target_area < 0.08:
                continue
            if overlap_w <= overlap_h:
                if (target_bounds["min_x"] + target_bounds["max_x"]) >= (anchor_bounds["min_x"] + anchor_bounds["max_x"]):
                    target_bounds["min_x"] = min(target_bounds["max_x"] - 4.0, target_bounds["min_x"] + overlap_w)
                else:
                    target_bounds["max_x"] = max(target_bounds["min_x"] + 4.0, target_bounds["max_x"] - overlap_w)
            else:
                if (target_bounds["min_y"] + target_bounds["max_y"]) >= (anchor_bounds["min_y"] + anchor_bounds["max_y"]):
                    target_bounds["min_y"] = min(target_bounds["max_y"] - 4.0, target_bounds["min_y"] + overlap_h)
                else:
                    target_bounds["max_y"] = max(target_bounds["min_y"] + 4.0, target_bounds["max_y"] - overlap_h)
            refined[target_idx]["x"] = round(target_bounds["min_x"], 1)
            refined[target_idx]["y"] = round(target_bounds["min_y"], 1)
            refined[target_idx]["width"] = round(target_bounds["max_x"] - target_bounds["min_x"], 1)
            refined[target_idx]["height"] = round(target_bounds["max_y"] - target_bounds["min_y"], 1)
            refined[target_idx]["polygon"] = _bbox_polygon(target_bounds)

    total_overlap_area = 0.0
    shared_wall_units = 0.0
    edge_count = 0
    adjacency = {room.get("id") or f"room-{index}": [] for index, room in enumerate(refined)}
    for index, room in enumerate(refined):
        room_id = room.get("id") or f"room-{index}"
        room["adjacent_rooms"] = []
        room["graph_degree"] = 0
        room["shared_wall_units"] = 0.0
        bounds = _room_bounds(room)
        label_x, label_y = _room_label_anchor(room, bounds)
        room["label_x"] = label_x
        room["label_y"] = label_y
        room["label_area_y"] = round(min(bounds["max_y"] - 1.4, label_y + 4.3), 1)
        room["bbox"] = bounds
        room["shape_confidence"] = round(min(0.94, max(0.48, 0.68 + min(bounds["width"], bounds["height"]) / max(bounds["width"], bounds["height"], 1) * 0.16)), 2)
    for left in range(len(refined)):
        for right in range(left + 1, len(refined)):
            left_bounds = _room_bounds(refined[left])
            right_bounds = _room_bounds(refined[right])
            _, _, overlap_area = _room_overlap_units(left_bounds, right_bounds)
            total_overlap_area += overlap_area
            shared = _shared_wall_units(left_bounds, right_bounds)
            if shared >= 3.0:
                left_id = refined[left].get("id") or f"room-{left}"
                right_id = refined[right].get("id") or f"room-{right}"
                adjacency[left_id].append(right_id)
                adjacency[right_id].append(left_id)
                refined[left]["shared_wall_units"] = round(float(refined[left].get("shared_wall_units") or 0) + shared, 2)
                refined[right]["shared_wall_units"] = round(float(refined[right].get("shared_wall_units") or 0) + shared, 2)
                shared_wall_units += shared
                edge_count += 1
    for room in refined:
        room_id = room.get("id")
        room["adjacent_rooms"] = sorted(set(adjacency.get(room_id, [])))
        room["graph_degree"] = len(room["adjacent_rooms"])
        overlap_penalty = 0.0
        bounds = room.get("bbox") or _room_bounds(room)
        room_area_units = max(bounds["width"] * bounds["height"], 1.0)
        for other in refined:
            if other.get("id") == room_id:
                continue
            _, _, overlap_area = _room_overlap_units(bounds, other.get("bbox") or _room_bounds(other))
            overlap_penalty += overlap_area / room_area_units
        room["overlap_ratio"] = round(min(1.0, overlap_penalty), 3)
        room["confidence"] = round(max(0.42, min(0.94, float(room.get("confidence") or 0.58) + min(room["graph_degree"], 3) * 0.03 - min(room["overlap_ratio"], 0.4) * 0.22)), 2)

    label_conflict_count = _deconflict_label_anchors(refined)

    envelope = {
        "min_x": min(room["bbox"]["min_x"] for room in refined),
        "max_x": max(room["bbox"]["max_x"] for room in refined),
        "min_y": min(room["bbox"]["min_y"] for room in refined),
        "max_y": max(room["bbox"]["max_y"] for room in refined),
    }
    envelope_width = max(0.01, envelope["max_x"] - envelope["min_x"])
    envelope_height = max(0.01, envelope["max_y"] - envelope["min_y"])
    envelope_area = envelope_width * envelope_height
    overlap_ratio = min(1.0, total_overlap_area / max(envelope_area, 1.0))
    connectivity_score = max(0.0, min(1.0, (edge_count / max(len(refined) - 1, 1)) * 0.7 + (1 - min(overlap_ratio * 3.4, 1)) * 0.3))
    return {
        "rooms": refined,
        "graph": {
            "edge_count": edge_count,
            "shared_wall_units": round(shared_wall_units, 2),
            "overlap_area_units": round(total_overlap_area, 2),
            "overlap_ratio": round(overlap_ratio, 3),
            "connectivity_score": round(connectivity_score, 3),
            "envelope_width_units": round(envelope_width, 2),
            "envelope_height_units": round(envelope_height, 2),
            "tightened_room_count": tightened_room_count,
            "label_conflict_count": label_conflict_count,
        },
    }


def _geometry_takeoff(rooms: list, built_up_area: float, carpet_area: float, wall_thickness_ft: float, dimension_evidence: dict | None = None, graph: dict | None = None, learning: dict | None = None):
    evidence = dimension_evidence or {}
    graph = graph if graph is not None else (_refine_rooms_with_wall_graph(rooms).get("graph", {}) if rooms else {})
    width = evidence.get("exterior_width_ft")
    depth = evidence.get("exterior_depth_ft")
    width_from_edge = False
    depth_from_edge = False
    envelope_width_units = float(graph.get("envelope_width_units") or 0)
    envelope_height_units = float(graph.get("envelope_height_units") or 0)
    envelope_aspect = float(evidence.get("aspect_hint") or 0) or (envelope_width_units / envelope_height_units if envelope_width_units and envelope_height_units else 1.28)
    envelope_aspect = max(0.55, min(2.4, envelope_aspect))
    source_prefix = str(evidence.get("source_prefix") or "ocr")
    calibrated_from_area = bool(evidence.get("calibrated_from_area"))
    edge_sums = evidence.get("edge_sums_ft") or {}
    edge_bound_width = _edge_bound_span(edge_sums, ("top", "bottom"))
    edge_bound_depth = _edge_bound_span(edge_sums, ("left", "right"))
    if not width and edge_bound_width:
        width = float(edge_bound_width)
        width_from_edge = True
    if not depth and edge_bound_depth:
        depth = float(edge_bound_depth)
        depth_from_edge = True
    if width and depth:
        external_wall_length = 2 * (float(width) + float(depth))
        if calibrated_from_area and source_prefix == "dxf":
            dimension_source = "dxf-envelope-area-calibrated"
        elif width_from_edge and depth_from_edge:
            dimension_source = f"{source_prefix}-edge-bound-graph"
        elif width_from_edge:
            dimension_source = f"{source_prefix}-edge-width-graph"
        elif depth_from_edge:
            dimension_source = f"{source_prefix}-edge-depth-graph"
        else:
            dimension_source = f"{source_prefix}-dimensions"
    elif width:
        width = float(width)
        depth_from_area = max(8.0, built_up_area / width)
        depth_from_graph = max(8.0, width / envelope_aspect)
        depth = round(depth_from_graph * 0.65 + depth_from_area * 0.35, 2)
        external_wall_length = 2 * (width + depth)
        dimension_source = f"{source_prefix}-edge-width-wall-graph" if width_from_edge else f"{source_prefix}-width-wall-graph"
    elif depth:
        depth = float(depth)
        width_from_area = max(8.0, built_up_area / depth)
        width_from_graph = max(8.0, depth * envelope_aspect)
        width = round(width_from_graph * 0.65 + width_from_area * 0.35, 2)
        external_wall_length = 2 * (width + depth)
        dimension_source = f"{source_prefix}-edge-depth-wall-graph" if depth_from_edge else f"{source_prefix}-depth-wall-graph"
    else:
        aspect = envelope_aspect
        width = math.sqrt(max(built_up_area, 1) * aspect)
        depth = built_up_area / width
        external_wall_length = 2 * (width + depth)
        dimension_source = f"{source_prefix}-wall-graph-area-derived" if rooms and source_prefix != "ocr" else ("wall-graph-area-derived" if rooms else "area-derived")

    raw_internal_units = sum(_polygon_perimeter_units(room) for room in rooms) * 0.5
    x_scale = float(width) / envelope_width_units if envelope_width_units else 0
    y_scale = float(depth) / envelope_height_units if envelope_height_units else 0
    active_scales = [scale for scale in (x_scale, y_scale) if scale]
    unit_scale = sum(active_scales) / len(active_scales) if active_scales else 0
    graph_internal_length = float(graph.get("shared_wall_units") or 0) * unit_scale if unit_scale else 0
    fallback_internal_length = raw_internal_units / max(125 + 92, 1) * external_wall_length * 1.7 if rooms else built_up_area * 0.12
    learned_internal_ratio = float((learning or {}).get("internal_wall_length_per_sqft") or 0)
    if 0.04 <= learned_internal_ratio <= 0.24:
        learned_internal_length = built_up_area * learned_internal_ratio
        fallback_internal_length = (fallback_internal_length * 0.55 + learned_internal_length * 0.45) if fallback_internal_length else learned_internal_length
    else:
        learned_internal_length = 0
    reported_shared_wall_length = graph_internal_length or (fallback_internal_length * 0.45 if rooms else 0)
    if graph_internal_length and fallback_internal_length:
        internal_wall_length = graph_internal_length * 0.7 + fallback_internal_length * 0.3
        internal_wall_source = "shared-wall-graph+learning" if learned_internal_length else "shared-wall-graph"
    else:
        internal_wall_length = graph_internal_length or learned_internal_length or fallback_internal_length
        if graph_internal_length:
            internal_wall_source = "shared-wall-graph"
        elif learned_internal_length:
            internal_wall_source = "learning-calibrated-fallback"
        else:
            internal_wall_source = "perimeter-fallback" if fallback_internal_length else "none"
    if not rooms:
        internal_wall_length = learned_internal_length or built_up_area * 0.12
        internal_wall_source = "learning-calibrated-fallback" if learned_internal_length else "area-fallback"
    internal_wall_length = max(built_up_area * 0.06, min(built_up_area * 0.22, internal_wall_length))
    wall_height = 10.0
    opening_factor = 0.16
    total_wall_length = external_wall_length + internal_wall_length
    gross_wall_area = total_wall_length * wall_height
    net_wall_area = gross_wall_area * (1 - opening_factor)
    brick_wall_area = net_wall_area * 0.62
    brick_volume_cft = brick_wall_area * wall_thickness_ft
    brick_count = brick_volume_cft * 13.5
    concrete_volume_cum = built_up_area * 0.018
    return {
        "dimension_source": dimension_source,
        "external_width_ft": round(width, 1),
        "external_depth_ft": round(depth, 1),
        "external_wall_length_ft": round(external_wall_length, 1),
        "internal_wall_length_ft": round(internal_wall_length, 1),
        "internal_wall_source": internal_wall_source,
        "shared_wall_length_ft": round(reported_shared_wall_length, 1) if reported_shared_wall_length else 0,
        "wall_graph_edges": int(graph.get("edge_count") or 0),
        "total_wall_length_ft": round(total_wall_length, 1),
        "wall_height_ft": wall_height,
        "opening_factor": opening_factor,
        "gross_wall_area_sqft": round(gross_wall_area, 1),
        "net_wall_area_sqft": round(net_wall_area, 1),
        "brickwork_area_sqft": round(brick_wall_area, 1),
        "brick_count": round(brick_count),
        "concrete_volume_cum": round(concrete_volume_cum, 2),
        "tile_area_sqft": round(carpet_area * 1.1, 1),
        "paint_area_sqft": round(net_wall_area * 1.65, 1),
    }


def _canonical_room_label(text: str):
    normalized = re.sub(r"[^A-Z0-9]+", " ", _normalize_arch_text(text))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None
    if re.search(r"\b(?:MASTER\s+BED\s+ROOM|BEDROOM|BED\s+ROOM|BEDRM)\b", normalized):
        return {"type": "bedroom", "label": "Bedroom"}
    if re.search(r"\b(?:BATH\s*ROOM|BATHROOM|TOILET|TOKET|TORET|T0ILET|WC)\b", normalized):
        return {"type": "bathroom", "label": "Bathroom"}
    if re.search(r"\b(?:KITCHEN|KIT)\b", normalized):
        return {"type": "kitchen", "label": "Kitchen"}
    if re.search(r"\b(?:LIVING|HALL|HALLWAY|DINING|SITTING|FAMILY|LOUNGE)\b", normalized):
        return {"type": "living", "label": "Living / Hall"}
    if re.search(r"\b(?:VERANDA|BALCONY|PORCH|TERRACE)\b", normalized):
        return {"type": "balcony", "label": "Balcony / Veranda"}
    if re.search(r"\b(?:STORE|STORAGE|UTILITY|UTILITIES|WASH|LAUNDRY|STAIR|STAR)\b", normalized):
        return {"type": "service", "label": "Service Zone"}
    return None


def _dedupe_label_boxes(label_boxes):
    filtered = []
    for box in label_boxes or []:
        text = str(box.get("text") or "")
        width = float(box.get("width") or 0)
        height = float(box.get("height") or 0)
        confidence = float(box.get("confidence") or 0)
        if confidence < 18:
            continue
        # Very wide OCR lines usually combine several labels, watermark text, or
        # dimension strings. They polluted the old model and created false rooms.
        if width > 34 or height > 9:
            continue
        if len(re.findall(r"\b(?:BED|BATH|TOILET|KITCHEN|HALL|ROOM|VERANDA|BALCONY|STORE|STAIR)\b", text.upper())) > 3:
            continue
        filtered.append(box)
    deduped = []
    for box in sorted(filtered, key=lambda item: item.get("confidence", 0), reverse=True):
        duplicate = False
        for existing in deduped:
            same_kind = existing["type"] == box["type"]
            close = math.hypot(existing["cx"] - box["cx"], existing["cy"] - box["cy"]) < 10.5
            overlapping_line = abs(existing["cy"] - box["cy"]) < 2.6 and abs(existing["cx"] - box["cx"]) < 18
            same_text = re.sub(r"[^A-Z0-9]+", "", str(existing.get("text", "")).upper()) == re.sub(r"[^A-Z0-9]+", "", str(box.get("text", "")).upper())
            if same_kind and (close or overlapping_line or (same_text and math.hypot(existing["cx"] - box["cx"], existing["cy"] - box["cy"]) < 8)):
                duplicate = True
                break
        if not duplicate:
            deduped.append(box)
    return deduped[:24]


def _spatial_label_hints(labels):
    hints = {
        "bedrooms": 0,
        "bathrooms": 0,
        "kitchens": 0,
        "halls": 0,
        "outdoor_zones": 0,
        "service_zones": 0,
        "stair_zones": 0,
        "primary_room_count": 0,
        "named_zone_count": 0,
        "total_area": None,
        "has_label_hint": False,
    }
    if not labels:
        return hints
    counters = {
        "bedroom": "bedrooms",
        "bathroom": "bathrooms",
        "kitchen": "kitchens",
        "living": "halls",
        "balcony": "outdoor_zones",
        "service": "service_zones",
    }
    for label in labels:
        key = counters.get(label.get("type"))
        if key:
            hints[key] += 1
    hints["primary_room_count"] = hints["bedrooms"] + hints["kitchens"] + hints["halls"] + hints["outdoor_zones"]
    hints["named_zone_count"] = hints["primary_room_count"] + hints["bathrooms"] + hints["service_zones"] + hints["stair_zones"]
    hints["has_label_hint"] = hints["named_zone_count"] > 0
    return hints


def _label_area_sqft(label):
    text = str(label.get("text") or "").upper()
    matches = [float(match) for match in re.findall(r"(\d{2,4}(?:\.\d+)?)\s*(?:SQ\s*FT|SQFT|SQ\.FT|SOFT)", text)]
    plausible = [value for value in matches if 20 <= value <= 600]
    return plausible[0] if plausible else None


def _label_box_defaults(room_type):
    return {
        "bedroom": (0.18, 0.16),
        "bathroom": (0.10, 0.10),
        "kitchen": (0.13, 0.12),
        "living": (0.24, 0.18),
        "balcony": (0.18, 0.10),
        "service": (0.12, 0.10),
    }.get(room_type, (0.14, 0.12))


def _label_component_limits(room_type):
    return {
        "bathroom": {"max_w": 0.22, "max_h": 0.22, "max_pixels": 0.040},
        "kitchen": {"max_w": 0.26, "max_h": 0.24, "max_pixels": 0.055},
        "service": {"max_w": 0.24, "max_h": 0.22, "max_pixels": 0.040},
        "bedroom": {"max_w": 0.30, "max_h": 0.30, "max_pixels": 0.085},
        "balcony": {"max_w": 0.34, "max_h": 0.24, "max_pixels": 0.060},
        "living": {"max_w": 0.42, "max_h": 0.36, "max_pixels": 0.120},
    }.get(room_type, {"max_w": 0.28, "max_h": 0.28, "max_pixels": 0.050})


def _room_label_for_type(room_type, count):
    if room_type == "bedroom":
        return f"Bedroom {count}"
    if room_type == "bathroom":
        return f"Bathroom {count}"
    if room_type == "living":
        return "Living / Hall" if count == 1 else f"Common Zone {count}"
    if room_type == "balcony":
        return "Balcony / Veranda" if count == 1 else f"Balcony / Veranda {count}"
    if room_type == "service":
        return "Service Zone" if count == 1 else f"Service Zone {count}"
    return "Kitchen"


def _point_to_free_pixel(free, px, py, radius=14):
    height, width = free.shape
    px = int(max(0, min(width - 1, round(px))))
    py = int(max(0, min(height - 1, round(py))))
    if free[py, px]:
        return px, py
    for distance in range(1, radius + 1):
        for dy in range(-distance, distance + 1):
            for dx in (-distance, distance):
                x = px + dx
                y = py + dy
                if 0 <= x < width and 0 <= y < height and free[y, x]:
                    return x, y
        for dx in range(-distance + 1, distance):
            for dy in (-distance, distance):
                x = px + dx
                y = py + dy
                if 0 <= x < width and 0 <= y < height and free[y, x]:
                    return x, y
    return None


def _component_bbox_for_point(labels, stats, component_id):
    if component_id <= 0 or component_id >= len(stats):
        return None
    x, y, width, height, area = stats[component_id]
    return {"bbox": (int(x), int(y), int(x + width - 1), int(y + height - 1)), "pixels": int(area)}


def _bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(1, int(max(0, x2 - x1) * max(0, y2 - y1)))


def _fallback_label_box(label, width, height):
    box_w_ratio, box_h_ratio = _label_box_defaults(label.get("type"))
    box_w = max(24, round(width * box_w_ratio))
    box_h = max(20, round(height * box_h_ratio))
    cx = float(label.get("cx", 62.5)) / 125 * width
    cy = float(label.get("cy", 46)) / 92 * height
    x1 = max(0, round(cx - box_w / 2))
    y1 = max(0, round(cy - box_h / 2))
    x2 = min(width - 1, x1 + box_w)
    y2 = min(height - 1, y1 + box_h)
    return {"bbox": (x1, y1, x2, y2), "pixels": max(1, (x2 - x1) * (y2 - y1))}


def _merge_pixel_bands(indices, max_gap=4):
    if not indices:
        return []
    bands = []
    start = prev = int(indices[0])
    for raw in indices[1:]:
        value = int(raw)
        if value - prev <= max_gap:
            prev = value
        else:
            bands.append((start, prev))
            start = prev = value
    bands.append((start, prev))
    return bands


def _dominant_wall_lines(wall_mask):
    if not cv2:
        return {"vertical": [], "horizontal": []}
    height, width = wall_mask.shape
    wall = (wall_mask > 0).astype(np.uint8) * 255
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, height // 18)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, width // 18), 1))
    vertical = cv2.morphologyEx(wall, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal = cv2.morphologyEx(wall, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    col_density = (vertical > 0).mean(axis=0)
    row_density = (horizontal > 0).mean(axis=1)
    col_threshold = max(0.08, float(np.percentile(col_density, 90)))
    row_threshold = max(0.08, float(np.percentile(row_density, 90)))
    vertical_bands = _merge_pixel_bands(np.where(col_density >= col_threshold)[0].tolist(), max_gap=max(3, width // 180))
    horizontal_bands = _merge_pixel_bands(np.where(row_density >= row_threshold)[0].tolist(), max_gap=max(3, height // 180))

    def centers(bands, limit):
        output = [0, limit - 1]
        for start, end in bands:
            if end - start < 1:
                continue
            output.append(round((start + end) / 2))
        return sorted(set(value for value in output if 0 <= value < limit))

    return {"vertical": centers(vertical_bands, width), "horizontal": centers(horizontal_bands, height)}


def _line_spans_inside_bbox(bbox, lines):
    x1, y1, x2, y2 = bbox
    vertical = [value for value in (lines.get("vertical") or []) if x1 + 1 < value < x2 - 1]
    horizontal = [value for value in (lines.get("horizontal") or []) if y1 + 1 < value < y2 - 1]
    return {"vertical": len(vertical), "horizontal": len(horizontal)}


def _fit_component_to_label(label, component, lines, width, height):
    if not component:
        return None
    room_type = label.get("type")
    x1, y1, x2, y2 = component["bbox"]
    box_w = max(1, x2 - x1 + 1)
    box_h = max(1, y2 - y1 + 1)
    box_pixels = max(1, int(component.get("pixels") or (box_w * box_h)))
    default_w_ratio, default_h_ratio = _label_box_defaults(room_type)
    limits = _label_component_limits(room_type)
    default_w = max(24, round(width * default_w_ratio))
    default_h = max(20, round(height * default_h_ratio))
    spans = _line_spans_inside_bbox(component["bbox"], lines)
    oversized = (
        box_w > max(default_w * 2.4, width * limits["max_w"])
        or box_h > max(default_h * 2.4, height * limits["max_h"])
        or box_pixels > max(default_w * default_h * 3.2, width * height * limits["max_pixels"])
        or spans["vertical"] >= 2
        or spans["horizontal"] >= 2
    )
    if not oversized:
        return component

    snapped = _snapped_label_box(label, lines, width, height)
    if snapped:
        sx1, sy1, sx2, sy2 = snapped["bbox"]
        snapped_w = max(1, sx2 - sx1 + 1)
        snapped_h = max(1, sy2 - sy1 + 1)
        if snapped_w <= box_w and snapped_h <= box_h:
            return {
                "bbox": snapped["bbox"],
                "pixels": snapped["pixels"],
                "source": snapped.get("source") or "wall-line-snap",
            }

    fallback = _fallback_label_box(label, width, height)
    fx1, fy1, fx2, fy2 = fallback["bbox"]
    fx1 = max(x1, fx1)
    fy1 = max(y1, fy1)
    fx2 = min(x2, fx2)
    fy2 = min(y2, fy2)
    if fx2 <= fx1 or fy2 <= fy1:
        fx1, fy1, fx2, fy2 = fallback["bbox"]
    return {
        "bbox": (int(fx1), int(fy1), int(fx2), int(fy2)),
        "pixels": _bbox_area((int(fx1), int(fy1), int(fx2), int(fy2))),
        "source": "label-fallback-fit",
    }


def _snapped_label_box(label, lines, width, height):
    vertical = lines.get("vertical") or []
    horizontal = lines.get("horizontal") or []
    if len(vertical) < 3 or len(horizontal) < 3:
        return None
    px = float(label.get("cx", 62.5)) / 125 * width
    py = float(label.get("cy", 46)) / 92 * height
    left_candidates = [value for value in vertical if value < px - 2]
    right_candidates = [value for value in vertical if value > px + 2]
    top_candidates = [value for value in horizontal if value < py - 2]
    bottom_candidates = [value for value in horizontal if value > py + 2]
    if not left_candidates or not right_candidates or not top_candidates or not bottom_candidates:
        return None
    left = max(left_candidates)
    right = min(right_candidates)
    top = max(top_candidates)
    bottom = min(bottom_candidates)
    box_w = right - left
    box_h = bottom - top
    if box_w < width * 0.045 or box_h < height * 0.045:
        return None
    if box_w > width * 0.42 or box_h > height * 0.40:
        return None
    max_size = {
        "bathroom": (0.22, 0.22),
        "kitchen": (0.24, 0.24),
        "service": (0.24, 0.22),
        "bedroom": (0.28, 0.30),
        "balcony": (0.32, 0.22),
        "living": (0.38, 0.34),
    }.get(label.get("type"), (0.30, 0.30))
    if box_w > width * max_size[0] or box_h > height * max_size[1]:
        return None

    # Expand once when the nearest line is a furniture/dimension line rather than a wall.
    room_type = label.get("type")
    min_ratio = {"bathroom": 0.06, "kitchen": 0.08, "bedroom": 0.12, "living": 0.15, "balcony": 0.10, "service": 0.06}.get(room_type, 0.08)
    if box_w < width * min_ratio and len(left_candidates) >= 2:
        left = left_candidates[-2]
    if right - left < width * min_ratio and len(right_candidates) >= 2:
        right = right_candidates[1]
    if box_h < height * min_ratio and len(top_candidates) >= 2:
        top = top_candidates[-2]
    if bottom - top < height * min_ratio and len(bottom_candidates) >= 2:
        bottom = bottom_candidates[1]

    margin = max(1, round(min(width, height) * 0.004))
    left = max(0, left + margin)
    top = max(0, top + margin)
    right = min(width - 1, right - margin)
    bottom = min(height - 1, bottom - margin)
    if right <= left or bottom <= top:
        return None
    return {"bbox": (int(left), int(top), int(right), int(bottom)), "pixels": int((right - left) * (bottom - top)), "source": "wall-line-snap"}


def _labels_to_anchored_rooms(image, labels, base_area, reprocess_attempt=0):
    if not cv2 or not labels:
        return []
    rgb = np.array(image.convert("RGB"))
    height, width, _ = rgb.shape
    if width < 80 or height < 80:
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    dark_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9)
    edges = cv2.Canny(gray, 60, 160)
    wall_mask = cv2.bitwise_or(dark_otsu, adaptive)
    wall_mask = cv2.bitwise_or(wall_mask, edges)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3 + min(reprocess_attempt, 2), 3 + min(reprocess_attempt, 2)))
    wall_mask = cv2.dilate(wall_mask, close_kernel, iterations=2 + min(reprocess_attempt, 2))
    dominant_lines = _dominant_wall_lines(wall_mask)

    free = wall_mask == 0
    num_labels, connected, stats, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), 8)
    image_area = width * height
    by_component = {}
    used_fallbacks = 0
    for label in labels:
        px = float(label.get("cx", 62.5)) / 125 * width
        py = float(label.get("cy", 46)) / 92 * height
        snapped = _snapped_label_box(label, dominant_lines, width, height)
        if snapped:
            key = f"snap-{label.get('type')}-{round(float(label.get('cx', 0)), 1)}-{round(float(label.get('cy', 0)), 1)}"
            bucket = by_component.setdefault(key, {"component": snapped, "labels": []})
            bucket["labels"].append(label)
            continue
        point = _point_to_free_pixel(free, px, py, radius=max(10, min(width, height) // 24))
        component = None
        if point:
            component_id = int(connected[point[1], point[0]])
            component = _component_bbox_for_point(connected, stats, component_id)
            component = _fit_component_to_label(label, component, dominant_lines, width, height)
            if component and component.get("source") == "label-fallback-fit":
                used_fallbacks += 1
        if not component or component["pixels"] > image_area * 0.08 or component["pixels"] < image_area * 0.001:
            component = _fallback_label_box(label, width, height)
            used_fallbacks += 1
            key = f"fallback-{label.get('type')}-{round(float(label.get('cx', 0)), 1)}-{round(float(label.get('cy', 0)), 1)}"
        else:
            component_id = int(connected[point[1], point[0]])
            key = f"component-{component_id}-{label.get('type')}-{round(float(label.get('cx', 0)) / 5) * 5}-{round(float(label.get('cy', 0)) / 5) * 5}"
        bucket = by_component.setdefault(key, {"component": component, "labels": []})
        bucket["labels"].append(label)

    rooms = []
    counters = {"bedroom": 0, "bathroom": 0, "kitchen": 0, "living": 0, "balcony": 0, "service": 0}
    total_component_pixels = sum(bucket["component"]["pixels"] for bucket in by_component.values()) or 1
    for bucket in sorted(by_component.values(), key=lambda item: min(label.get("cy", 0) for label in item["labels"])):
        best_label = max(bucket["labels"], key=lambda item: (float(item.get("confidence") or 0), -float(item.get("width") or 0)))
        room_type = best_label.get("type") or "service"
        counters[room_type] = counters.get(room_type, 0) + 1
        count = counters[room_type]
        x1, y1, x2, y2 = bucket["component"]["bbox"]
        bbox_width = max(1, x2 - x1 + 1)
        bbox_height = max(1, y2 - y1 + 1)
        explicit_area = _label_area_sqft(best_label)
        area_ratio = bucket["component"]["pixels"] / total_component_pixels
        room_area = explicit_area or max(28, float(base_area) * 0.82 * area_ratio)
        nx1 = round(x1 / width * 125, 1)
        ny1 = round(y1 / height * 92, 1)
        nx2 = round(x2 / width * 125, 1)
        ny2 = round(y2 / height * 92, 1)
        rooms.append({
            "id": f"ocr-anchor-{len(rooms) + 1}",
            "type": room_type,
            "label": _room_label_for_type(room_type, count),
            "x": nx1,
            "y": ny1,
            "width": round(max(2, bbox_width / width * 125), 1),
            "height": round(max(2, bbox_height / height * 92), 1),
            "area_sqft": round(room_area, 1),
            "confidence": round(min(0.9, max(0.58, float(best_label.get("confidence") or 45) / 100)), 2),
            "polygon": [[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]],
            "label_x": round(float(best_label.get("cx") or (nx1 + nx2) / 2), 1),
            "label_y": round(float(best_label.get("cy") or (ny1 + ny2) / 2), 1),
        "source": bucket["component"].get("source") or "ocr-anchored-cv",
            "ocr_label_text": best_label.get("text", ""),
        })
    total_area = sum(float(room["area_sqft"]) for room in rooms)
    target_area = max(1, float(base_area) * 0.82)
    if total_area and not any(_label_area_sqft(label) for label in labels):
        scale = target_area / total_area
        for room in rooms:
            room["area_sqft"] = round(float(room["area_sqft"]) * scale, 1)
    for room in rooms:
        if used_fallbacks:
            room["confidence"] = round(max(0.52, float(room["confidence"]) - 0.08), 2)
    return rooms[:18]


def _line_text_separated_image(gray):
    if not cv2:
        return None
    arr = np.array(gray)
    binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    height, width = binary.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, width // 22), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, height // 22)))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    line_mask = cv2.dilate(cv2.bitwise_or(horizontal_lines, vertical_lines), np.ones((2, 2), dtype=np.uint8), iterations=1)
    no_lines = arr.copy()
    no_lines[line_mask > 0] = 255
    return Image.fromarray(no_lines)


def _graphic_layer_suppressed_image(gray):
    if not cv2:
        return None
    arr = np.array(gray)
    binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.full_like(arr, 255)
    image_area = arr.shape[0] * arr.shape[1]
    for label_id in range(1, num_labels):
        x, y, width, height, area = stats[label_id]
        if area < 4:
            continue
        aspect = width / max(height, 1)
        # Keep components shaped like text/dimensions; suppress large furniture,
        # hatch fills, symbols, and heavy graphic clusters for OCR-only passes.
        if area > image_area * 0.018:
            continue
        if width > arr.shape[1] * 0.18 or height > arr.shape[0] * 0.12:
            continue
        if 0.08 <= aspect <= 18:
            cleaned[labels == label_id] = arr[labels == label_id]
    return Image.fromarray(cleaned)


def _perimeter_dimension_focus_image(gray):
    arr = np.array(gray)
    if arr.ndim != 2:
        return None
    height, width = arr.shape
    band_y = max(18, round(height * 0.16))
    band_x = max(18, round(width * 0.16))
    focused = np.full_like(arr, 255)
    focused[:band_y, :] = arr[:band_y, :]
    focused[max(0, height - band_y) :, :] = arr[max(0, height - band_y) :, :]
    focused[:, :band_x] = np.minimum(focused[:, :band_x], arr[:, :band_x])
    focused[:, max(0, width - band_x) :] = np.minimum(focused[:, max(0, width - band_x) :], arr[:, max(0, width - band_x) :])
    return Image.fromarray(focused)


def _prepare_ocr_images(image):
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(3.0, max(1.0, 1600 / max(width, height)))
    if scale > 1:
        rgb = rgb.resize((round(width * scale), round(height * scale)))
    gray = rgb.convert("L")
    if ImageOps:
        gray = ImageOps.autocontrast(gray)
    if ImageEnhance:
        gray = ImageEnhance.Contrast(gray).enhance(1.7)
        gray = ImageEnhance.Sharpness(gray).enhance(1.35)
    if ImageFilter:
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
    arr = np.array(gray)
    thresholds = [int(np.percentile(arr, 58))]
    variants = [gray]
    perimeter = _perimeter_dimension_focus_image(gray)
    if perimeter:
        variants.append(perimeter)
    no_lines = _line_text_separated_image(gray)
    if no_lines:
        variants.append(no_lines)
    text_layer = _graphic_layer_suppressed_image(gray)
    if text_layer:
        variants.append(text_layer)
    for threshold in thresholds:
        binary = Image.fromarray(np.where(arr < threshold, 0, 255).astype(np.uint8))
        variants.append(binary)

    # Crop only the obvious blank border for extra OCR passes while keeping full-image passes
    # for spatial label matching.
    dark = arr < min(225, int(np.percentile(arr, 72)))
    ys, xs = np.where(dark)
    if len(xs) and len(ys):
        pad = round(max(arr.shape) * 0.025)
        left = max(0, int(xs.min()) - pad)
        top = max(0, int(ys.min()) - pad)
        right = min(arr.shape[1], int(xs.max()) + pad)
        bottom = min(arr.shape[0], int(ys.max()) + pad)
        if right - left > arr.shape[1] * 0.35 and bottom - top > arr.shape[0] * 0.35:
            cropped = gray.crop((left, top, right, bottom))
            variants.append(cropped)
    variants.append(gray.rotate(90, expand=True, fillcolor=255))
    return variants[:5]


def _easyocr_floorplan_labels(image):
    if not easyocr or os.getenv("ENABLE_EASYOCR", "0") != "1":
        return {"text": "", "labels": [], "confidences": []}
    global EASYOCR_READER
    try:
        if EASYOCR_READER is None:
            EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
        rgb = np.array(image.convert("RGB"))
        results = EASYOCR_READER.readtext(rgb, detail=1, paragraph=False)
    except Exception:
        return {"text": "", "labels": [], "confidences": []}
    width, height = image.size
    labels = []
    texts = []
    confidences = []
    for box, text, confidence in results:
        clean = str(text or "").strip()
        if not clean:
            continue
        score = float(confidence or 0) * 100
        clean_norm = _normalize_arch_text(clean)
        texts.append(clean_norm)
        confidences.append(score)
        canonical = _canonical_room_label(clean_norm)
        if not canonical or score < 22:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        labels.append({
            **canonical,
            "text": clean_norm,
            "confidence": round(score, 1),
            "cx": round(((left + right) / 2) / width * 125, 2),
            "cy": round(((top + bottom) / 2) / height * 92, 2),
            "x": round(left / width * 125, 2),
            "y": round(top / height * 92, 2),
            "width": round((right - left) / width * 125, 2),
            "height": round((bottom - top) / height * 92, 2),
            "engine": "easyocr",
        })
    return {"text": " ".join(texts), "labels": labels, "confidences": confidences}


def _ocr_floorplan_text(plan, reprocess_attempt=0):
    cache_key = str(int(reprocess_attempt or 0))
    ocr_cache = plan.setdefault("_ocr_cache", {})
    if cache_key in ocr_cache:
        return ocr_cache[cache_key]
    if plan["file_type"] == "pdf":
        text = _normalize_arch_text((plan.get("pdf_meta") or {}).get("text") or "")
        hints = _label_hints_from_text(text)
        result = {
            "text": text[:800],
            "avg_confidence": 88 if text else 0,
            "word_count": len(re.findall(r"[A-Z0-9./'\"]+", text)),
            "hints": hints,
            "dimension_evidence": _extract_dimension_evidence(text),
            "available": True,
            "labels": [],
            "dimension_boxes": [],
            "source": "pdf-vector-text" if text else "pdf-raster-no-text",
        }
        ocr_cache[cache_key] = result
        return result
    if not pytesseract or not Image or plan["file_type"] not in {"png", "jpg", "jpeg"}:
        return {"text": "", "avg_confidence": 0, "word_count": 0, "hints": {}, "dimension_evidence": {}, "available": bool(pytesseract)}
    try:
        image = Image.open(plan["path"])
    except Exception:
        return {"text": "", "avg_confidence": 0, "word_count": 0, "hints": {}, "dimension_evidence": {}, "available": True}

    configs = ["--oem 3 --psm 11"] if reprocess_attempt == 0 else ["--oem 3 --psm 11", "--oem 3 --psm 6"]
    prepared_images = _prepare_ocr_images(image)
    if reprocess_attempt == 0:
        prepared_images = prepared_images[:4]
    full_width, full_height = prepared_images[0].size if prepared_images else image.size
    best = {"text": "", "avg_confidence": 0, "word_count": 0, "hints": {}, "dimension_evidence": {}, "available": True, "labels": [], "dimension_boxes": []}
    collected_text = []
    collected_confidences = []
    label_boxes = []
    dimension_boxes = []
    for prepared in prepared_images:
        for config in configs:
            try:
                data = pytesseract.image_to_data(prepared, lang="eng", config=config, output_type=pytesseract.Output.DICT)
            except Exception:
                continue
            words = []
            confidences = []
            line_groups = {}
            for idx, (text, conf) in enumerate(zip(data.get("text", []), data.get("conf", []))):
                token = str(text or "").strip()
                try:
                    score = float(conf)
                except Exception:
                    score = -1
                if token and score >= 8:
                    words.append(token)
                    confidences.append(score)
                    group_key = (
                        data.get("block_num", [0] * len(data.get("text", [])))[idx],
                        data.get("par_num", [0] * len(data.get("text", [])))[idx],
                        data.get("line_num", [0] * len(data.get("text", [])))[idx],
                    )
                    left = float(data.get("left", [0])[idx])
                    top = float(data.get("top", [0])[idx])
                    width = float(data.get("width", [0])[idx])
                    height = float(data.get("height", [0])[idx])
                    group = line_groups.setdefault(group_key, {"tokens": [], "conf": [], "left": left, "top": top, "right": left + width, "bottom": top + height})
                    group["tokens"].append(token)
                    group["conf"].append(score)
                    group["left"] = min(group["left"], left)
                    group["top"] = min(group["top"], top)
                    group["right"] = max(group["right"], left + width)
                    group["bottom"] = max(group["bottom"], top + height)
            joined = _normalize_arch_text(" ".join(words))
            if joined:
                collected_text.append(joined)
                collected_confidences.extend(confidences)
            if prepared.size == (full_width, full_height):
                for group in line_groups.values():
                    line_text = _normalize_arch_text(" ".join(group["tokens"]))
                    avg_line_conf = sum(group["conf"]) / max(len(group["conf"]), 1)
                    cx = ((group["left"] + group["right"]) / 2) / full_width * 125
                    cy = ((group["top"] + group["bottom"]) / 2) / full_height * 92
                    edge_distances = {
                        "top": cy,
                        "bottom": 92 - cy,
                        "left": cx,
                        "right": 125 - cx,
                    }
                    nearest_edge = min(edge_distances, key=edge_distances.get)
                    if edge_distances[nearest_edge] > 16:
                        nearest_edge = "interior"
                    for token in re.findall(r"\b\d{1,3}\s*(?:'|-)\s*\d{1,2}\s*(?:\"|IN)?|\b\d{1,3}\s*'\s*|\b\d{1,3}\s*FT\b", line_text.upper()):
                        value = _dimension_token_to_ft(token)
                        if value and 2 <= value <= 80 and avg_line_conf >= 8:
                            dimension_boxes.append({
                                "text": line_text,
                                "token": token,
                                "value_ft": round(value, 2),
                                "confidence": round(avg_line_conf, 1),
                                "edge": nearest_edge,
                                "cx": round(cx, 2),
                                "cy": round(cy, 2),
                                "x": round(group["left"] / full_width * 125, 2),
                                "y": round(group["top"] / full_height * 92, 2),
                            })
                    canonical = _canonical_room_label(line_text)
                    if not canonical:
                        continue
                    if avg_line_conf < 10:
                        continue
                    label_boxes.append({
                        **canonical,
                        "text": line_text,
                        "confidence": round(avg_line_conf, 1),
                        "cx": round(cx, 2),
                        "cy": round(cy, 2),
                        "x": round(group["left"] / full_width * 125, 2),
                        "y": round(group["top"] / full_height * 92, 2),
                        "width": round((group["right"] - group["left"]) / full_width * 125, 2),
                        "height": round((group["bottom"] - group["top"]) / full_height * 92, 2),
                    })
            hints = _label_hints_from_text(joined)
            avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0
            candidate_score = (10 if hints.get("has_label_hint") else 0) + len(words) + avg_conf / 10
            best_score = (10 if best["hints"].get("has_label_hint") else 0) + best["word_count"] + best["avg_confidence"] / 10
            if candidate_score > best_score:
                best = {
                    "text": joined,
                    "avg_confidence": avg_conf,
                    "word_count": len(words),
                    "hints": hints,
                    "dimension_evidence": _extract_dimension_evidence(joined, dimension_boxes=dimension_boxes),
                    "labels": label_boxes[:],
                    "dimension_boxes": dimension_boxes[:],
                    "available": True,
                }
    easy = _easyocr_floorplan_labels(image)
    if easy.get("text"):
        collected_text.append(_normalize_arch_text(easy["text"]))
        collected_confidences.extend(easy.get("confidences") or [])
        label_boxes.extend(easy.get("labels") or [])
    if collected_text:
        combined = _normalize_arch_text(" ".join(collected_text))
        tokens = re.findall(r"[A-Za-z0-9./]+", combined)
        combined_hints = _label_hints_from_text(combined)
        deduped_labels = _dedupe_label_boxes(label_boxes)
        spatial_hints = _spatial_label_hints(deduped_labels)
        if best["hints"].get("total_area") and not combined_hints.get("total_area"):
            combined_hints["total_area"] = best["hints"]["total_area"]
        combined_hints = _merge_label_hints(spatial_hints, best["hints"], combined_hints)
        if spatial_hints.get("has_label_hint"):
            for key in ("bathrooms", "kitchens", "halls", "outdoor_zones", "service_zones", "stair_zones"):
                if spatial_hints.get(key):
                    combined_hints[key] = min(int(combined_hints.get(key) or 0), int(spatial_hints[key]) + (1 if key == "halls" else 0))
            if spatial_hints.get("bedrooms"):
                combined_hints["bedrooms"] = max(int(spatial_hints["bedrooms"]), min(int(combined_hints.get("bedrooms") or 0), int(spatial_hints["bedrooms"]) + 2))
            combined_hints["primary_room_count"] = combined_hints["bedrooms"] + combined_hints["kitchens"] + combined_hints["halls"] + combined_hints["outdoor_zones"]
            combined_hints["named_zone_count"] = combined_hints["primary_room_count"] + combined_hints["bathrooms"] + combined_hints["service_zones"] + combined_hints["stair_zones"]
        best = {
            "text": " ".join(tokens[:80]),
            "avg_confidence": round(max(best["avg_confidence"], sum(collected_confidences) / len(collected_confidences) if collected_confidences else 0), 1),
            "word_count": len(tokens),
            "hints": combined_hints,
            "dimension_evidence": _extract_dimension_evidence(combined, None, dimension_boxes),
            "labels": deduped_labels,
            "dimension_boxes": _dedupe_dimension_boxes(dimension_boxes),
            "available": True,
        }
    ocr_cache[cache_key] = best
    return best


def _raster_room_detection(plan, base_area, reprocess_attempt=0, labels=None):
    if not Image:
        return None
    try:
        source_image = Image.open(plan["path"]).convert("RGB")
        image = source_image.convert("L")
    except Exception:
        return None

    original_width, original_height = image.size
    max_dim = 720
    scale = min(max_dim / max(original_width, original_height), 1)
    if scale < 1:
        image = image.resize((round(original_width * scale), round(original_height * scale)))
    arr = np.array(image)
    height, width = arr.shape
    image_area = width * height
    if image_area < 10_000:
        return {
            "rooms": [],
            "quality_note": "Uploaded raster is too small for reliable room segmentation.",
            "polygon_completeness": 0.28,
            "geometry_consistency": 0.35,
            "detected_room_count": 0,
            "wall_density": 0,
        }

    threshold_percentiles = [28, 22, 38, 48]
    dilation_iterations = [2, 1, 3, 4]
    threshold = min(210, max(105, int(np.percentile(arr, threshold_percentiles[min(reprocess_attempt, 3)]))))
    dark = arr < threshold
    wall_density = float(dark.mean())
    label_anchor_rooms = _labels_to_anchored_rooms(source_image.resize(image.size), labels or [], base_area, reprocess_attempt)
    color_rooms = _color_region_rooms(image, base_area)
    line_rooms = _line_projection_rooms(dark, base_area)
    # Reprocessing increases dilation to close door gaps and broken walls.
    walls = _binary_dilate(dark, iterations=dilation_iterations[min(reprocess_attempt, 3)])
    free = ~walls

    # Remove exterior/background free space by flood filling from image edges.
    edge_seed = np.zeros_like(free, dtype=bool)
    edge_seed[0, :] = free[0, :]
    edge_seed[-1, :] = free[-1, :]
    edge_seed[:, 0] = free[:, 0]
    edge_seed[:, -1] = free[:, -1]
    exterior = np.zeros_like(free, dtype=bool)
    queue = deque([(x, y) for y, x in zip(*np.where(edge_seed))])
    for x, y in queue:
        exterior[y, x] = True
    while queue:
        cx, cy = queue.popleft()
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if 0 <= nx < width and 0 <= ny < height and free[ny, nx] and not exterior[ny, nx]:
                exterior[ny, nx] = True
                queue.append((nx, ny))

    interior = free & ~exterior
    min_pixel_ratios = [0.003, 0.0018, 0.004, 0.006]
    min_pixels = max(90, int(image_area * min_pixel_ratios[min(reprocess_attempt, 3)]))
    max_pixels = int(image_area * 0.45)
    components = [
        component
        for component in _connected_components(interior, min_pixels)
        if component["pixels"] <= max_pixels
    ]
    components.sort(key=lambda item: item["pixels"], reverse=True)
    components = components[:14]

    if len(components) <= 1:
        # Fallback: segment enclosed wall blocks when exterior flood-fill fails on open plans.
        free_components = _connected_components(free, min_pixels)
        free_components.sort(key=lambda item: item["pixels"], reverse=True)
        components = [item for item in free_components[1:15] if item["pixels"] <= max_pixels]

    contour_rooms = _classify_boxes_as_rooms(components, base_area, width, height, "contour")
    candidates = [label_anchor_rooms, color_rooms, line_rooms, contour_rooms]
    rooms = max(candidates, key=lambda candidate: _rooms_score(candidate, wall_density))

    bbox_coverage = sum((room["width"] * room["height"]) for room in rooms) / (125 * 92) if rooms else 0
    oversegmentation_penalty = max(0, len(rooms) - 10) * 0.025
    polygon_completeness = max(0.24, min(0.76, 0.34 + min(len(rooms), 10) * 0.045 + bbox_coverage * 0.18 - oversegmentation_penalty))
    geometry_consistency = max(0.30, min(0.78, 0.48 + min(len(rooms), 8) * 0.035 - abs(wall_density - 0.18) * 0.55 - oversegmentation_penalty))
    return {
        "rooms": rooms,
        "quality_note": "Raster segmentation used OCR-anchored wall regions when labels were available, with contours as fallback.",
        "polygon_completeness": polygon_completeness,
        "geometry_consistency": geometry_consistency,
        "detected_room_count": len(rooms),
        "wall_density": round(wall_density, 3),
        "oversegmented": len(rooms) > 10,
        "detector": rooms[0].get("source", "contour") if rooms else "none",
        "ocr_anchor_count": len(label_anchor_rooms),
    }


def detect_layout(plan_id: str, project_area: float | None = None, reprocess_attempt: int = 0):
    plan = _get_plan(plan_id)
    dxf = plan.get("dxf_meta") or {}
    learning = _learning_profile(plan)
    ocr = _ocr_floorplan_text(plan, reprocess_attempt)
    label_hints = _merge_label_hints(plan.get("label_hints") or {}, ocr.get("hints") or {})
    label_hints = _apply_learning_hints(label_hints, learning)
    dimension_evidence = ocr.get("dimension_evidence") or {}
    if dimension_evidence.get("area_from_text_sqft"):
        label_hints["total_area"] = max(float(label_hints.get("total_area") or 0), float(dimension_evidence["area_from_text_sqft"]))
    base_area = _choose_plan_area(dxf, label_hints, dimension_evidence, project_area)
    dimension_evidence = _merge_dimension_evidence(dimension_evidence, _dxf_dimension_evidence(dxf, base_area))
    raster = _raster_room_detection(plan, base_area, reprocess_attempt, ocr.get("labels") or []) if plan["file_type"] in {"png", "jpg", "jpeg"} else None
    label_override = False
    segmented_room_count = 0
    raw_zone_count = 0
    count_source = "geometry"
    graph_metrics = {
        "edge_count": 0,
        "shared_wall_units": 0.0,
        "overlap_ratio": 0.0,
        "connectivity_score": 0.0,
        "envelope_width_units": 0.0,
        "envelope_height_units": 0.0,
        "tightened_room_count": 0,
        "label_conflict_count": 0,
    }
    service_zone_count = int(label_hints.get("service_zones") or 0) + int(label_hints.get("stair_zones") or 0)
    if raster and raster["rooms"]:
        rooms = _apply_ocr_labels_to_rooms(raster["rooms"], ocr.get("labels") or [])
        rooms = _augment_rooms_from_hints(rooms, base_area, label_hints)
        refined = _refine_rooms_with_wall_graph(rooms)
        rooms = refined["rooms"]
        graph_metrics = refined["graph"]
        segmented_room_count = len(rooms)
        raw_zone_count = len(rooms)
        segment_counts = _segment_count_summary(rooms)
        count_source = segment_counts["count_source"]
        room_count = len(rooms)
        bedroom_count = sum(1 for room in rooms if room["type"] == "bedroom")
        bathroom_count = sum(1 for room in rooms if room["type"] == "bathroom")
        kitchen_count = sum(1 for room in rooms if room["type"] == "kitchen")
        hall_count = sum(1 for room in rooms if room["type"] == "living")
        service_zone_count = max(service_zone_count, segment_counts["service"])
        if label_hints.get("has_label_hint"):
            label_primary_rooms = int(label_hints.get("primary_room_count") or 0)
            bedroom_count = label_hints.get("bedrooms") or bedroom_count
            bathroom_count = label_hints.get("bathrooms") or bathroom_count
            kitchen_count = label_hints.get("kitchens") or kitchen_count
            hall_count = (label_hints.get("halls") or 0) + (label_hints.get("outdoor_zones") or 0) or hall_count
            service_zone_count = int(label_hints.get("service_zones") or 0) + int(label_hints.get("stair_zones") or 0)
            if label_primary_rooms >= 2:
                # Text labels are a stronger signal than raw contours on watermarked plans.
                room_count = label_primary_rooms
                label_override = True
                count_source = "ocr-label-fusion"
            else:
                room_count = max(segment_counts["primary"], bedroom_count + kitchen_count + hall_count)
        else:
            room_count = segment_counts["primary"]
            bedroom_count = min(bedroom_count, room_count)
            bathroom_count = max(bathroom_count, segment_counts["wet"])
            hall_count = max(0, room_count - bedroom_count - kitchen_count)
        if label_override and ocr.get("labels") and raster.get("detector") not in {"ocr-anchored-cv", "wall-line-snap"}:
            label_rooms = _rooms_from_ocr_labels(ocr.get("labels") or [], base_area, label_hints)
            if len(label_rooms) >= max(3, min(room_count, 6)):
                refined = _refine_rooms_with_wall_graph(label_rooms)
                rooms = refined["rooms"]
                graph_metrics = refined["graph"]
    elif dxf.get("labels"):
        cad_counts = _counts_from_hints_and_dxf(label_hints, dxf)
        bedroom_count = cad_counts["bedroom"]
        bathroom_count = cad_counts["bathroom"]
        kitchen_count = cad_counts["kitchen"]
        hall_count = cad_counts["living"]
        service_zone_count = cad_counts["service"]
        outdoor_zone_count = cad_counts["balcony"]
        room_count = bedroom_count + kitchen_count + hall_count + outdoor_zone_count
        raw_zone_count = room_count + bathroom_count + service_zone_count
        count_source = "cad-labels"
    else:
        room_count = 0
        bedroom_count = 0
        bathroom_count = 0
        kitchen_count = 0
        hall_count = 0
        raw_zone_count = 0
        count_source = "none"

    if not (raster and raster["rooms"]):
        if plan.get("demo"):
            rooms = _demo_rooms(base_area)
        elif dxf.get("labels"):
            rooms = _rooms_from_dxf_text_entities(dxf, base_area, label_hints)
        else:
            rooms = []
        refined = _refine_rooms_with_wall_graph(rooms)
        rooms = refined["rooms"]
        graph_metrics = refined["graph"]
        if rooms:
            bedroom_count = sum(1 for room in rooms if room["type"] == "bedroom")
            bathroom_count = sum(1 for room in rooms if room["type"] == "bathroom")
            kitchen_count = sum(1 for room in rooms if room["type"] == "kitchen")
            hall_count = sum(1 for room in rooms if room["type"] == "living")
            outdoor_zone_count = sum(1 for room in rooms if room["type"] == "balcony")
            service_zone_count = sum(1 for room in rooms if room["type"] == "service")
            room_count = bedroom_count + kitchen_count + hall_count + outdoor_zone_count
            raw_zone_count = len(rooms)
    wall_thickness_ft = float(learning.get("wall_thickness_ft") or (0.5 if plan["file_type"] != "dxf" else 0.45))
    wall_thickness_ft = round(max(0.35, min(wall_thickness_ft, 0.9)), 2)
    built_up_area = round(base_area * (1.0 + (0.02 if plan["file_type"] == "dxf" else 0)), 1)
    carpet_area = round(built_up_area * 0.78, 1)
    polygon_area = round(sum(room["area_sqft"] for room in rooms) * 1.08, 1)
    geometry_takeoff = _geometry_takeoff(rooms, built_up_area, carpet_area, wall_thickness_ft, dimension_evidence, graph_metrics, learning)
    variance = abs(polygon_area - built_up_area) / built_up_area if built_up_area else 0

    quality_score = plan["quality"]["score"]
    if raster:
        raw_ocr_score = float(ocr.get("avg_confidence") or 0) / 100
        ocr_confidence = max(0.18, min(0.72, raw_ocr_score)) if ocr.get("word_count") else 0.18
        if label_hints.get("has_label_hint"):
            ocr_confidence = max(ocr_confidence, 0.48)
        if label_override:
            ocr_confidence = max(ocr_confidence, 0.62)
        polygon_completeness = raster["polygon_completeness"]
        geometry_consistency = min(0.92, max(raster["geometry_consistency"], float(graph_metrics.get("connectivity_score") or 0) * 0.88))
    else:
        ocr_confidence = 0.86 if dxf.get("labels") else 0.20
        polygon_completeness = min(0.94, 0.70 + min(len(rooms), 8) * 0.025 + (0.08 if plan["file_type"] == "dxf" else 0))
        geometry_consistency = max(0.35, min(0.94, 0.91 - variance + float(graph_metrics.get("connectivity_score") or 0) * 0.08))
    has_detected_dimensions = bool(
        dxf.get("has_dimensions")
        or label_hints.get("total_area")
        or dimension_evidence.get("dimension_count", 0) >= 2
        or float(dimension_evidence.get("confidence") or 0) >= 45
    )
    dimension_confidence = max(0.12, min(1.0, float(dimension_evidence.get("confidence") or 0) / 100)) if has_detected_dimensions else 0.0
    graph_confidence = max(
        0.18,
        min(
            1.0,
            float(graph_metrics.get("connectivity_score") or 0)
            - min(int(graph_metrics.get("label_conflict_count") or 0), 3) * 0.04
            - min(int(graph_metrics.get("tightened_room_count") or 0), 4) * 0.015,
        ),
    )
    overlap_ratio = float(graph_metrics.get("overlap_ratio") or 0)

    recommendations = []
    if raster:
        recommendations.append("Raster image was processed using OCR-anchored wall detection, with contour segmentation only as fallback.")
        recommendations.append(f"Detection mode: {raster.get('detector', 'contour')} segmentation.")
        recommendations.append("OCR preprocessing used binary contrast, line-text separation, graphic-layer suppression, rotated text passes, and architecture-format cleanup.")
        if raster.get("ocr_anchor_count"):
            recommendations.append(f"OCR anchors mapped {raster['ocr_anchor_count']} named zone(s) back to nearby wall geometry.")
        if ocr.get("word_count"):
            recommendations.append(f"Tesseract OCR read {ocr['word_count']} text token(s) at about {ocr['avg_confidence']}% average confidence.")
        else:
            recommendations.append("Tesseract OCR did not find reliable room labels in this image.")
        if label_hints.get("has_label_hint"):
            recommendations.append("OCR or filename label hints were used to improve room classification. Confirm counts before pricing.")
        if label_override and segmented_room_count and segmented_room_count != room_count:
            recommendations.append(f"OCR label model corrected the primary room count from {segmented_room_count} segmented zones to {room_count} named rooms.")
        if not label_override and segmented_room_count and segmented_room_count != room_count:
            recommendations.append(f"Raw geometry found {segmented_room_count} possible zones, but only {room_count} were treated as primary rooms. Confirm this before estimating.")
        if raster["detected_room_count"] <= 1:
            recommendations.append("Room boundaries were not sufficiently enclosed. Use reprocess or upload a clearer plan with darker wall lines.")
        if raster.get("oversegmented"):
            recommendations.append("The raster appears over-segmented into many small zones. Treat room count as provisional and confirm manually.")
        if reprocess_attempt:
            recommendations.append(f"Reprocess attempt {reprocess_attempt} used an alternate wall and contour sensitivity.")
    if dxf.get("total_area") and project_area:
        project_variance = abs(float(project_area) - float(dxf["total_area"])) / float(project_area)
        if project_variance > 0.08:
            recommendations.append(f"CAD area label shows {round(dxf['total_area'])} sq.ft, which differs from the project area by {round(project_variance * 100, 1)}%. Confirm the selected flat scope.")
    if not has_detected_dimensions:
        recommendations.append("Dimensions were not reliably detected. Confirm the built-up area before final pricing.")
    else:
        if dimension_evidence.get("dimension_count", 0):
            recommendations.append(f"Dimension pattern engine found {dimension_evidence.get('dimension_count', 0)} dimension value(s); takeoff source is {geometry_takeoff['dimension_source'].replace('-', ' ')}.")
        else:
            recommendations.append(f"Vector geometry calibrated the envelope span; takeoff source is {geometry_takeoff['dimension_source'].replace('-', ' ')}.")
    if variance > 0.06:
        recommendations.append(f"Detected area differs from polygon-derived area by {round(variance * 100, 1)}%. Manual verification is recommended.")
    if overlap_ratio > 0.025:
        recommendations.append(f"Wall-graph cleanup still found {round(overlap_ratio * 100, 1)}% overlapping zone area. Review corridor and open-space boundaries.")
    if int(graph_metrics.get("label_conflict_count") or 0) > 0:
        recommendations.append("Adjacent OCR labels still cluster in nearby wall cells. Review bedroom and bathroom names once.")
    if graph_confidence < 0.55:
        recommendations.append("Shared-wall graph is still sparse, so internal wall takeoff should be treated as provisional.")
    if polygon_completeness < 0.82:
        recommendations.append("Some wall boundaries look open or disconnected, so the layout should be reviewed once.")
    if bathroom_count >= 2 and ocr_confidence < 0.78:
        recommendations.append("Low OCR confidence around wet-area labels. Bathroom count should be confirmed.")
    if not recommendations:
        recommendations.append("Layout geometry is consistent enough for a demo-grade flat-wise estimate.")

    confidence = _score_confidence(
        {
            "quality": quality_score / 100,
            "ocr": ocr_confidence,
            "polygon": polygon_completeness,
            "geometry": geometry_consistency,
            "dimensions": dimension_confidence,
            "graph": graph_confidence,
            "overlap": overlap_ratio,
            "missing_dimensions": 0 if has_detected_dimensions else 1,
            "manual_corrections": min(1, int(learning.get("corrections_applied") or 0) / 4),
        }
    )
    if raster:
        if label_override:
            confidence = min(max(confidence, 58), 72 if 3 <= room_count <= 10 else 62)
            if dimension_evidence.get("confidence", 0) >= 55 and 3 <= room_count <= 12:
                confidence = max(confidence, 70)
        else:
            confidence = min(confidence, 64 if label_hints.get("has_label_hint") and 3 <= room_count <= 10 else (58 if 3 <= room_count <= 10 else 48))

    result = {
        "plan_id": plan_id,
        "rooms": rooms,
        "summary": {
            "room_count": room_count,
            "primary_room_count": room_count,
            "raw_zone_count": raw_zone_count,
            "wet_area_count": bathroom_count,
            "service_zone_count": service_zone_count,
            "count_source": count_source,
            "bedroom_count": bedroom_count,
            "bathroom_count": bathroom_count,
            "kitchen_count": kitchen_count,
            "hall_count": hall_count,
            "wall_thickness_ft": wall_thickness_ft,
            "carpet_area_sqft": carpet_area,
            "built_up_area_sqft": built_up_area,
            "polygon_area_sqft": polygon_area,
            **geometry_takeoff,
            "doors": max(0, room_count + 1) if room_count else 0,
            "windows": max(0, room_count + 2) if room_count else 0,
        },
        "signals": {
            "ocr_confidence": round(ocr_confidence * 100),
            "polygon_completeness": round(polygon_completeness * 100),
            "geometry_consistency": round(geometry_consistency * 100),
            "image_quality": quality_score,
            "area_variance_percent": round(variance * 100, 1),
            "segmented_zone_count": segmented_room_count,
            "named_zone_count": label_hints.get("named_zone_count") or 0,
            "dimension_confidence": dimension_evidence.get("confidence", 0),
            "wall_graph_confidence": round(graph_confidence * 100),
            "wall_graph_edges": int(graph_metrics.get("edge_count") or 0),
            "wall_graph_overlap_percent": round(overlap_ratio * 100, 1),
            "wall_graph_tightened_rooms": int(graph_metrics.get("tightened_room_count") or 0),
            "label_conflicts": int(graph_metrics.get("label_conflict_count") or 0),
            "takeoff_confidence": (
                80
                if geometry_takeoff["dimension_source"] == "ocr-edge-bound-graph"
                else (
                82
                if geometry_takeoff["dimension_source"] == "ocr-dimensions"
                else (
                    78
                    if geometry_takeoff["dimension_source"].startswith("dxf-")
                    else (74 if geometry_takeoff["dimension_source"].startswith("ocr-") else (68 if "wall-graph" in geometry_takeoff["dimension_source"] else 62))
                )
                )
            ),
        },
        "confidence": confidence,
        "confidence_explanation": _confidence_explanation(confidence, plan, variance, dxf, dimension_confidence, graph_confidence, overlap_ratio),
        "recommendations": recommendations,
        "reprocess_attempt": reprocess_attempt,
        "ocr": {
            "available": ocr.get("available", False),
            "word_count": ocr.get("word_count", 0),
            "avg_confidence": ocr.get("avg_confidence", 0),
            "text": ocr.get("text", "")[:600],
            "spatial_labels": ocr.get("labels", [])[:16],
            "dimension_evidence": dimension_evidence,
            "processing_features": [
                "binarization",
                "line_text_separation",
                "graphic_layer_suppression",
                "rotated_text_passes",
                "symbol_normalization",
                "contextual_ocr_correction",
                "dictionary_format_matching",
                "dimension_wall_clustering",
                "pdf_vector_text_extraction" if plan["file_type"] == "pdf" else "raster_ocr",
            ],
        },
    }
    plan["detection"] = result
    _persist_plan(plan)
    return result


def confirm_layout(plan_id: str, summary: dict, rooms: list | None = None, user_corrections: int = 0):
    plan = _get_plan(plan_id)
    detection = plan.get("detection") or detect_layout(plan_id)
    if rooms:
        relabeled = relabel_edited_zones(plan_id, rooms, summary, user_corrections, persist=False)
        manual_summary = {
            key: value
            for key, value in (summary or {}).items()
            if key in {"wall_thickness_ft", "carpet_area_sqft", "built_up_area_sqft", "external_wall_length_ft", "internal_wall_length_ft", "total_wall_length_ft", "wall_height_ft", "net_wall_area_sqft"}
        }
        updated = {**relabeled, "summary": {**relabeled["summary"], **manual_summary}}
    else:
        updated = {**detection, "summary": {**detection["summary"], **summary}}
    penalty = min(user_corrections * 3, 12)
    updated["confidence"] = max(52, updated["confidence"] - penalty)
    if user_corrections:
        updated["confidence_explanation"] = f"Confidence adjusted to {updated['confidence']}% because {user_corrections} field(s) were manually corrected."
    updated["confirmed"] = True
    _update_learning_from_layout(plan, updated.get("rooms") or detection.get("rooms") or [], updated.get("summary") or detection.get("summary") or {}, user_corrections)
    plan["confirmed"] = updated
    _persist_plan(plan)
    return updated


def _clamp_zone(room: dict, base_area: float):
    raw_polygon = room.get("polygon") or []
    polygon = []
    for point in raw_polygon:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        polygon.append([
            round(max(0, min(float(point[0]), 125)), 1),
            round(max(0, min(float(point[1]), 92)), 1),
        ])
    if len(polygon) < 3:
        x = max(0, min(float(room.get("x", 8)), 121))
        y = max(0, min(float(room.get("y", 8)), 88))
        width = max(4, min(float(room.get("width", 18)), 125 - x))
        height = max(4, min(float(room.get("height", 14)), 92 - y))
        polygon = [[round(x, 1), round(y, 1)], [round(x + width, 1), round(y, 1)], [round(x + width, 1), round(y + height, 1)], [round(x, 1), round(y + height, 1)]]
    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    width = max(4, max_x - min_x)
    height = max(4, max_y - min_y)
    polygon_units = 0
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        polygon_units += point[0] * next_point[1] - next_point[0] * point[1]
    polygon_units = abs(polygon_units) / 2
    area_sqft = round(max(20, (polygon_units / (125 * 92)) * base_area * 1.08), 1)
    border_thickness_ft = max(0.25, min(float(room.get("border_thickness_ft") or room.get("wall_thickness_ft") or 0.5), 2.0))
    return {
        **room,
        "id": str(room.get("id") or f"zone-{uuid.uuid4().hex[:6]}"),
        "x": round(min_x, 1),
        "y": round(min_y, 1),
        "width": round(width, 1),
        "height": round(height, 1),
        "border_thickness_ft": round(border_thickness_ft, 2),
        "area_sqft": area_sqft,
        "polygon": polygon,
    }


def _learning_profile(plan: dict):
    learning = plan.setdefault(
        "learning",
        {
            "corrections_applied": 0,
            "wall_thickness_ft": None,
            "room_type_counts": {},
            "average_room_area_sqft": {},
            "internal_wall_length_per_sqft": None,
            "dimension_source_counts": {},
        },
    )
    learning.setdefault("corrections_applied", 0)
    learning.setdefault("wall_thickness_ft", None)
    learning.setdefault("room_type_counts", {})
    learning.setdefault("average_room_area_sqft", {})
    learning.setdefault("internal_wall_length_per_sqft", None)
    learning.setdefault("dimension_source_counts", {})
    return learning


def _apply_learning_hints(label_hints: dict, learning: dict):
    if not learning or not int(learning.get("corrections_applied") or 0):
        return label_hints
    room_type_counts = learning.get("room_type_counts") or {}
    enriched = dict(label_hints or {})
    enriched["bedrooms"] = max(int(enriched.get("bedrooms") or 0), int(room_type_counts.get("bedroom") or 0))
    enriched["bathrooms"] = max(int(enriched.get("bathrooms") or 0), int(room_type_counts.get("bathroom") or 0))
    enriched["kitchens"] = max(int(enriched.get("kitchens") or 0), int(room_type_counts.get("kitchen") or 0))
    enriched["halls"] = max(int(enriched.get("halls") or 0), int(room_type_counts.get("living") or 0))
    enriched["outdoor_zones"] = max(int(enriched.get("outdoor_zones") or 0), int(room_type_counts.get("balcony") or 0))
    enriched["service_zones"] = max(int(enriched.get("service_zones") or 0), int(room_type_counts.get("service") or 0))
    if int(learning.get("corrections_applied") or 0) >= 2 and room_type_counts:
        learned_map = {
            "bedrooms": int(room_type_counts.get("bedroom") or 0),
            "bathrooms": int(room_type_counts.get("bathroom") or 0),
            "kitchens": int(room_type_counts.get("kitchen") or 0),
            "halls": int(room_type_counts.get("living") or 0),
            "outdoor_zones": int(room_type_counts.get("balcony") or 0),
            "service_zones": int(room_type_counts.get("service") or 0),
        }
        for key, learned in learned_map.items():
            if not learned:
                continue
            observed = int(enriched.get(key) or 0)
            tolerance = 1 if key in {"bathrooms", "halls"} else 0
            enriched[key] = max(learned, min(observed or learned, learned + tolerance))
    enriched["primary_room_count"] = enriched["bedrooms"] + enriched["kitchens"] + enriched["halls"] + enriched["outdoor_zones"]
    enriched["named_zone_count"] = enriched["primary_room_count"] + enriched["bathrooms"] + enriched["service_zones"] + int(enriched.get("stair_zones") or 0)
    enriched["has_label_hint"] = enriched["named_zone_count"] > 0 or bool(enriched.get("has_label_hint"))
    return enriched


def _update_learning_from_layout(plan: dict, rooms: list, summary: dict, user_corrections: int = 0):
    learning = _learning_profile(plan)
    learning["corrections_applied"] = int(learning.get("corrections_applied") or 0) + max(0, int(user_corrections or 0))
    wall_thickness = summary.get("wall_thickness_ft")
    if wall_thickness:
        wall_thickness = float(wall_thickness)
        current = learning.get("wall_thickness_ft")
        learning["wall_thickness_ft"] = round((float(current) * 0.55 + wall_thickness * 0.45) if current else wall_thickness, 2)
    counts = {}
    area_totals = {}
    for room in rooms or []:
        room_type = str(room.get("type") or "")
        if not room_type or room_type == "unlabeled":
            continue
        counts[room_type] = counts.get(room_type, 0) + 1
        area_totals.setdefault(room_type, []).append(float(room.get("area_sqft") or 0))
    if counts:
        learning["room_type_counts"] = counts
    if area_totals:
        learning["average_room_area_sqft"] = {
            room_type: round(sum(values) / len([value for value in values if value > 0]), 1)
            for room_type, values in area_totals.items()
            if any(value > 0 for value in values)
        }
    built_up_area = float(summary.get("built_up_area_sqft") or 0)
    internal_wall_length = float(summary.get("internal_wall_length_ft") or 0)
    if built_up_area > 0 and internal_wall_length > 0:
        observed_ratio = max(0.04, min(0.24, internal_wall_length / built_up_area))
        current_ratio = learning.get("internal_wall_length_per_sqft")
        learning["internal_wall_length_per_sqft"] = round((float(current_ratio) * 0.6 + observed_ratio * 0.4) if current_ratio else observed_ratio, 4)
    dimension_source = str(summary.get("dimension_source") or "").strip()
    if dimension_source:
        source_counts = dict(learning.get("dimension_source_counts") or {})
        source_counts[dimension_source] = int(source_counts.get(dimension_source) or 0) + 1
        learning["dimension_source_counts"] = source_counts
    return learning


def _expected_room_areas(learning: dict | None = None):
    expected = {
        "bathroom": 60,
        "kitchen": 110,
        "bedroom": 130,
        "living": 240,
        "balcony": 90,
        "service": 55,
    }
    learned_areas = (learning or {}).get("average_room_area_sqft") or {}
    for room_type, value in learned_areas.items():
        if room_type in expected and value:
            expected[room_type] = max(24, min(420, float(value)))
    return expected


def _assign_zone_labels(zones: list, target_counts: dict, learning: dict | None = None):
    if not zones:
        return []
    labeled = []
    remaining = set()
    expected_area = _expected_room_areas(learning)
    for index, zone in enumerate(zones):
        if zone.get("user_corrected_label"):
            room_type = str(zone.get("type") or "unlabeled")
            room_label = str(zone.get("label") or room_type.replace("_", " ").title())
            labeled.append({**zone, "type": room_type, "label": room_label, "confidence": max(float(zone.get("confidence") or 0.58), 0.86), "label_source": "user-corrected"})
        else:
            labeled.append({**zone, "type": "unlabeled", "label": "Zone pending AI label", "confidence": 0.58, "label_source": "ai-pending"})
            remaining.add(index)

    manual_counts = {}
    for zone in labeled:
        if zone.get("label_source") == "user-corrected":
            manual_counts[zone["type"]] = manual_counts.get(zone["type"], 0) + 1

    def take(predicate=None, sort_key=None, reverse=False):
        candidates = [idx for idx in remaining if predicate is None or predicate(labeled[idx])]
        if not candidates:
            return None
        if sort_key:
            candidates.sort(key=lambda idx: sort_key(labeled[idx]), reverse=reverse)
        idx = candidates[0]
        remaining.remove(idx)
        return idx

    kitchen_target = max(0, min(max(int(target_counts.get("kitchens") or 0), 0), 1) - manual_counts.get("kitchen", 0))
    bath_target = max(0, min(max(int(target_counts.get("bathrooms") or 0), 0), max(0, len(labeled) - 1)) - manual_counts.get("bathroom", 0))
    bed_target = max(0, min(max(int(target_counts.get("bedrooms") or 0), 0), max(0, len(labeled) - kitchen_target - bath_target)) - manual_counts.get("bedroom", 0))

    largest_idx = take(sort_key=lambda zone: zone["area_sqft"], reverse=True)
    if largest_idx is not None and manual_counts.get("living", 0) == 0:
        labeled[largest_idx].update({"type": "living", "label": "Living / Hall", "confidence": 0.76})
    elif largest_idx is not None:
        remaining.add(largest_idx)

    if kitchen_target:
        kitchen_idx = take(
            predicate=lambda zone: zone["area_sqft"] >= 45,
            sort_key=lambda zone: (
                abs(float(zone["area_sqft"]) - expected_area["kitchen"]),
                abs((zone["x"] + zone["width"] / 2) - 92),
                abs((zone["y"] + zone["height"] / 2) - 42),
            ),
        )
        if kitchen_idx is not None:
            labeled[kitchen_idx].update({"type": "kitchen", "label": "Kitchen", "confidence": 0.74, "label_source": "ai-rechecked"})

    for index in range(bath_target):
        bath_idx = take(sort_key=lambda zone: (abs(float(zone["area_sqft"]) - expected_area["bathroom"]), float(zone["area_sqft"])))
        if bath_idx is not None:
            labeled[bath_idx].update({"type": "bathroom", "label": f"Bathroom {manual_counts.get('bathroom', 0) + index + 1}", "confidence": 0.72, "label_source": "ai-rechecked"})

    for index in range(bed_target):
        bed_idx = take(
            predicate=lambda zone: zone["area_sqft"] >= 55,
            sort_key=lambda zone: (abs(float(zone["area_sqft"]) - expected_area["bedroom"]), -zone["y"], -zone["width"]),
        )
        if bed_idx is None:
            bed_idx = take(sort_key=lambda zone: abs(float(zone["area_sqft"]) - expected_area["bedroom"]))
        if bed_idx is not None:
            labeled[bed_idx].update({"type": "bedroom", "label": f"Bedroom {manual_counts.get('bedroom', 0) + index + 1}", "confidence": 0.73, "label_source": "ai-rechecked"})

    service_index = 1
    for idx in list(remaining):
        zone = labeled[idx]
        if zone["area_sqft"] < 65 or zone["width"] < 9 or zone["height"] < 9:
            zone.update({"type": "service", "label": f"Service Zone {manual_counts.get('service', 0) + service_index}", "confidence": 0.62, "label_source": "ai-rechecked"})
            service_index += 1
        else:
            zone.update({"type": "living", "label": f"Common Zone {manual_counts.get('living', 0) + service_index}", "confidence": 0.64, "label_source": "ai-rechecked"})
            service_index += 1
    return labeled


def relabel_edited_zones(plan_id: str, rooms: list, summary: dict | None = None, user_corrections: int = 1, persist: bool = True):
    plan = _get_plan(plan_id)
    detection = plan.get("detection") or detect_layout(plan_id)
    learning = _learning_profile(plan)
    base_summary = {**detection.get("summary", {}), **(summary or {})}
    base_area = float(base_summary.get("built_up_area_sqft") or base_summary.get("polygon_area_sqft") or 1250)
    cleaned = [_clamp_zone(room, base_area) for room in rooms if float(room.get("width", 0) or 0) > 1 and float(room.get("height", 0) or 0) > 1]
    if not cleaned:
        cleaned = detection.get("rooms", [])
    target_counts = {
        "bedrooms": base_summary.get("bedroom_count") or 0,
        "bathrooms": base_summary.get("bathroom_count") or 0,
        "kitchens": base_summary.get("kitchen_count") or 0,
    }
    labeled_rooms = _assign_zone_labels(cleaned, target_counts, learning)
    refined = _refine_rooms_with_wall_graph(labeled_rooms)
    labeled_rooms = refined["rooms"]
    graph_metrics = refined["graph"]
    bedroom_count = sum(1 for room in labeled_rooms if room["type"] == "bedroom")
    bathroom_count = sum(1 for room in labeled_rooms if room["type"] == "bathroom")
    kitchen_count = sum(1 for room in labeled_rooms if room["type"] == "kitchen")
    hall_count = sum(1 for room in labeled_rooms if room["type"] == "living")
    service_zone_count = sum(1 for room in labeled_rooms if room["type"] == "service")
    room_count = bedroom_count + kitchen_count + hall_count
    polygon_area = round(sum(room["area_sqft"] for room in labeled_rooms) * 1.08, 1)
    wall_thickness = round(sum(float(room.get("border_thickness_ft", 0.5)) for room in labeled_rooms) / max(len(labeled_rooms), 1), 2)
    built_up_area = round(float(base_summary.get("built_up_area_sqft") or polygon_area), 1)
    carpet_area = round(float(base_summary.get("carpet_area_sqft") or built_up_area * 0.78), 1)
    geometry_takeoff = _geometry_takeoff(labeled_rooms, built_up_area, carpet_area, wall_thickness, (detection.get("ocr") or {}).get("dimension_evidence") or {}, graph_metrics, learning)
    variance = abs(polygon_area - built_up_area) / built_up_area if built_up_area else 0
    confidence = max(52, min(86, int(detection.get("confidence", 58)) + 8 - min(user_corrections * 2, 10) - round(min(variance, 0.25) * 20)))
    recommendations = list(detection.get("recommendations", []))
    manual_label_count = sum(1 for room in cleaned if room.get("user_corrected_label"))
    recommendations.insert(0, "Edited zones were rechecked using corrected geometry, OCR counts, room size, and layout context.")
    if manual_label_count:
        recommendations.insert(1, f"{manual_label_count} user-corrected zone label(s) were preserved during AI re-check.")
    else:
        recommendations.insert(1, "User-adjusted boundaries improved geometry alignment; labels were rechecked by AI.")
    recommendations = list(dict.fromkeys(recommendations))
    updated = {
        **detection,
        "rooms": labeled_rooms,
        "summary": {
            **base_summary,
            "room_count": room_count,
            "primary_room_count": room_count,
            "raw_zone_count": len(labeled_rooms),
            "wet_area_count": bathroom_count,
            "service_zone_count": service_zone_count,
            "count_source": "user-geometry-auto-label",
            "bedroom_count": bedroom_count,
            "bathroom_count": bathroom_count,
            "kitchen_count": kitchen_count,
            "hall_count": hall_count,
            "wall_thickness_ft": wall_thickness,
            "carpet_area_sqft": carpet_area,
            "built_up_area_sqft": built_up_area,
            "polygon_area_sqft": polygon_area,
            **geometry_takeoff,
            "doors": max(0, room_count + bathroom_count + 1),
            "windows": max(0, room_count + 2),
        },
        "signals": {
            **detection.get("signals", {}),
            "polygon_completeness": max(detection.get("signals", {}).get("polygon_completeness", 0), 84),
            "geometry_consistency": max(detection.get("signals", {}).get("geometry_consistency", 0), max(82, round(float(graph_metrics.get("connectivity_score") or 0) * 100))),
            "area_variance_percent": round(variance * 100, 1),
            "segmented_zone_count": len(labeled_rooms),
            "wall_graph_confidence": max(detection.get("signals", {}).get("wall_graph_confidence", 0), round(float(graph_metrics.get("connectivity_score") or 0) * 100)),
            "wall_graph_edges": int(graph_metrics.get("edge_count") or 0),
            "wall_graph_overlap_percent": round(float(graph_metrics.get("overlap_ratio") or 0) * 100, 1),
            "wall_graph_tightened_rooms": int(graph_metrics.get("tightened_room_count") or 0),
            "label_conflicts": int(graph_metrics.get("label_conflict_count") or 0),
        },
        "confidence": confidence,
        "confidence_explanation": f"Confidence is {confidence}% because user-corrected boundaries were auto-labeled by the system and area variance is {round(variance * 100, 1)}%.",
        "recommendations": recommendations[:8],
        "edited_geometry": True,
    }
    if persist:
        _update_learning_from_layout(plan, labeled_rooms, updated["summary"], user_corrections)
        plan["detection"] = updated
        _persist_plan(plan)
    return updated


def estimate_flat(plan_id: str, rate_per_sqft: float, tier: str = "Standard"):
    plan = _get_plan(plan_id)
    layout = plan.get("confirmed") or plan.get("detection") or detect_layout(plan_id)
    summary = layout["summary"]
    area = float(summary["built_up_area_sqft"])
    carpet_area = float(summary.get("carpet_area_sqft") or area * 0.78)
    wall_length = float(summary.get("total_wall_length_ft") or area * 0.18)
    net_wall_area = float(summary.get("net_wall_area_sqft") or wall_length * float(summary.get("wall_height_ft") or 10) * 0.84)
    tile_area = float(summary.get("tile_area_sqft") or carpet_area * 1.10)
    paint_area = float(summary.get("paint_area_sqft") or net_wall_area * 1.65)
    brick_count = float(summary.get("brick_count") or net_wall_area * 4.2)
    concrete_volume = float(summary.get("concrete_volume_cum") or area * 0.018)
    tier_multiplier = FLAT_FINISH_FACTORS.get(tier, 1.0)
    applied_rate = float(rate_per_sqft) * tier_multiplier
    area_estimate = round(area * applied_rate)

    material_quantities = [
        {"material": "Steel", "base_quantity": max(area * 3.6, concrete_volume * 105), "unit": "kg", "rate": MATERIAL_RATES["steel"], "formula": "max(Area x 3.6 kg/sqft, RCC concrete x 105 kg/cum)"},
        {"material": "Concrete", "base_quantity": concrete_volume, "unit": "cum", "rate": MATERIAL_RATES["concrete"], "formula": "Built-up area x 0.018 cum/sqft"},
        {"material": "Cement", "base_quantity": area * 0.28 + net_wall_area * 0.055, "unit": "bags", "rate": MATERIAL_RATES["cement"], "formula": "Slab/plaster allowance + wall area x 0.055 bags/sqft"},
        {"material": "Sand", "base_quantity": area * 0.85 + net_wall_area * 0.22, "unit": "cft", "rate": MATERIAL_RATES["sand"], "formula": "Floor/plaster allowance + wall area x 0.22 cft/sqft"},
        {"material": "Bricks", "base_quantity": brick_count, "unit": "nos", "rate": MATERIAL_RATES["bricks"], "formula": "Net brickwork wall area x wall thickness x 13.5 bricks/cft"},
        {"material": "Tiles", "base_quantity": tile_area, "unit": "sqft", "rate": MATERIAL_RATES["tiles"], "formula": "Carpet area x 1.10"},
        {"material": "Paint", "base_quantity": paint_area, "unit": "sqft", "rate": MATERIAL_RATES["paint"], "formula": "Net wall area x 1.65 coats/sides multiplier"},
    ]
    materials = []
    for row in material_quantities:
        waste_factor = MATERIAL_WASTE_FACTORS.get(row["material"], 0)
        quantity = round(row["base_quantity"] * (1 + waste_factor), 1)
        amount = round(quantity * row["rate"])
        materials.append({**row, "quantity": quantity, "waste_factor": waste_factor, "approx_cost": amount})

    direct_material_cost = sum(row["approx_cost"] for row in materials)
    material_cost_share = 0.45
    material_based_estimate = round(direct_material_cost / material_cost_share)
    labour_cost = round(area_estimate * 0.22)
    finishing_cost = round(area_estimate * 0.24)
    services_cost = round(area_estimate * 0.14)
    variance = abs(area_estimate - material_based_estimate) / area_estimate if area_estimate else 0
    confidence = max(50, min(96, layout["confidence"] - round(max(0, variance - 0.08) * 45)))
    if layout.get("confidence", 0) >= 70 and variance <= 0.18:
        confidence = max(confidence, 70)
    if layout.get("confidence", 0) >= 58 and variance <= 0.08:
        confidence = max(confidence, 70)

    result = {
        "plan_id": plan_id,
        "tier": tier,
        "rate_per_sqft": round(applied_rate, 2),
        "area_based_estimate": area_estimate,
        "material_based_estimate": material_based_estimate,
        "total_estimated_cost": round((area_estimate * 0.72) + (material_based_estimate * 0.28)),
        "estimated_material_cost": direct_material_cost,
        "labour_cost": labour_cost,
        "finishing_cost": finishing_cost,
        "services_cost": services_cost,
        "variance_percent": round(variance * 100, 1),
        "confidence": confidence,
        "confidence_explanation": _estimate_confidence_text(confidence, variance, layout),
        "area_details": layout["summary"],
        "material_breakdown": materials,
        "formula_trace": {
            "source": "construction_estimation_formulas.pdf",
            "area_based": "C_base = Area x Regional Rate x Finish Factor",
            "cad_pipeline": "Dimensions/OCR -> Wall length -> Area -> Quantity -> Material -> Price",
            "wall_takeoff": "Wall area = (external + internal wall length) x wall height x (1 - openings)",
            "waste": "Q_effective = Q_base x (1 + Waste Percentage)",
            "finish_factor": tier_multiplier,
            "material_cost_share_assumption": material_cost_share,
        },
        "cost_distribution": [
            {"name": "Material", "value": direct_material_cost},
            {"name": "Labour", "value": labour_cost},
            {"name": "Finishing", "value": finishing_cost},
            {"name": "MEP", "value": services_cost},
        ],
        "recommendations": _final_recommendations(layout, variance),
    }
    plan["flat_estimate"] = result
    return result


def _score_confidence(signals: dict):
    score = (
        signals["quality"] * 16
        + signals["ocr"] * 18
        + signals["polygon"] * 16
        + signals["geometry"] * 16
        + signals.get("dimensions", 0) * 18
        + signals.get("graph", 0) * 10
        + (1 - min(signals.get("overlap", 0), 0.35) / 0.35) * 8
        + (1 - signals["missing_dimensions"]) * 4
        + (1 - min(signals["manual_corrections"], 1)) * 4
    )
    if signals["missing_dimensions"]:
        score -= 7
    return round(max(40, min(96, score)))


def _confidence_explanation(confidence: int, plan: dict, variance: float, dxf: dict, dimension_confidence: float = 0, graph_confidence: float = 0, overlap_ratio: float = 0):
    reasons = []
    if plan["quality"]["score"] < 75:
        reasons.append("image clarity is moderate")
    if dimension_confidence < 0.45 and not dxf.get("has_dimensions"):
        reasons.append("dimensions were partially missing")
    if graph_confidence < 0.55:
        reasons.append("wall connectivity is still provisional")
    if overlap_ratio > 0.025:
        reasons.append("some detected zones still overlap")
    if variance > 0.06:
        reasons.append("polygon area and project area differ slightly")
    if not reasons:
        return f"Confidence is {confidence}% because room labels, wall graph continuity, and area checks are consistent."
    return f"Confidence is {confidence}% because " + ", ".join(reasons) + "."


def _estimate_confidence_text(confidence: int, variance: float, layout: dict):
    if variance < 0.08 and confidence >= 70:
        return f"Confidence is {confidence}% because area-based and material-based estimates are closely aligned."
    return f"Confidence is {confidence}% because estimate variance is {round(variance * 100, 1)}% and layout confidence is {layout['confidence']}%."


def _final_recommendations(layout: dict, variance: float):
    recommendations = list(layout.get("recommendations", []))
    if variance > 0.12:
        recommendations.append("Material model varies materially from area pricing. Recheck rate and structural assumptions.")
    if str(layout["summary"].get("dimension_source") or "").endswith("area-derived"):
        recommendations.append("Takeoff still fell back to graph or area-derived dimensions. Validate one known span before procurement.")
    if layout["summary"].get("wall_thickness_ft", 0) < 0.45:
        recommendations.append("Wall thickness appears lean. Confirm masonry and structural wall assumptions before procurement.")
    if layout["summary"].get("bathroom_count", 0) >= 2:
        recommendations.append("Wet-area waterproofing and plumbing allowances should be validated in the BOQ.")
    return recommendations[:5]
