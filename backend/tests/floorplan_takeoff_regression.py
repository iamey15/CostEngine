import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.flat_plan_engine import _apply_learning_hints, _choose_plan_area, _demo_rooms, _dxf_dimension_evidence, _extract_dimension_evidence, _fit_component_to_label, _geometry_takeoff, _refine_rooms_with_wall_graph


def _ft_label(value):
    feet = int(value)
    inches = round((value - feet) * 12)
    return f"{feet}'-{inches}\"" if inches else f"{feet}'"


def _dimension_box(value, edge, axis):
    if edge in {"top", "bottom"}:
        return {
            "text": _ft_label(value),
            "token": _ft_label(value),
            "value_ft": value,
            "confidence": 90,
            "edge": edge,
            "cx": axis,
            "cy": 3 if edge == "top" else 89,
        }
    return {
        "text": _ft_label(value),
        "token": _ft_label(value),
        "value_ft": value,
        "confidence": 90,
        "edge": edge,
        "cx": 3 if edge == "left" else 122,
        "cy": axis,
    }


def run_regression(seed=42, cases=100):
    random.seed(seed)
    failures = []
    for index in range(cases):
        width = random.uniform(24, 62)
        depth = random.uniform(18, 48)
        area = width * depth
        top_segments = [width * random.uniform(0.18, 0.34), width * random.uniform(0.18, 0.34)]
        top_segments.append(width - sum(top_segments))
        left_segments = [depth * random.uniform(0.25, 0.45), depth * random.uniform(0.18, 0.35)]
        left_segments.append(depth - sum(left_segments))
        boxes = []
        axis = 15
        for value in top_segments:
            boxes.append(_dimension_box(round(value, 2), "top", axis))
            axis += 30
        axis = 18
        for value in left_segments:
            boxes.append(_dimension_box(round(value, 2), "left", axis))
            axis += 24
        text = " ".join(box["text"] for box in boxes) + f" {round(area)} SQ FT"
        evidence = _extract_dimension_evidence(text, area, boxes)
        takeoff = _geometry_takeoff([], area, area * 0.78, 0.5, evidence)
        width_error = abs(takeoff["external_width_ft"] - width) / width
        depth_error = abs(takeoff["external_depth_ft"] - depth) / depth
        wall_error = abs(takeoff["external_wall_length_ft"] - 2 * (width + depth)) / (2 * (width + depth))
        if width_error > 0.08 or depth_error > 0.08 or wall_error > 0.08 or evidence["confidence"] < 70:
            failures.append(
                {
                    "case": index,
                    "width_error": round(width_error, 3),
                    "depth_error": round(depth_error, 3),
                    "wall_error": round(wall_error, 3),
                    "confidence": evidence["confidence"],
                    "source": takeoff["dimension_source"],
                }
            )
    return {"cases": cases, "passed": cases - len(failures), "failed": len(failures), "failures": failures[:10]}


def run_partial_dimension_regression():
    rooms = _demo_rooms(1250)
    top_only = [_dimension_box(31.0, "top", 24), _dimension_box(10.0, "top", 56)]
    left_only = [_dimension_box(18.0, "left", 18), _dimension_box(22.0, "left", 50)]

    top_evidence = _extract_dimension_evidence("31' 10' 1250 SQ FT", 1250, top_only)
    left_evidence = _extract_dimension_evidence("18' 22' 1250 SQ FT", 1250, left_only)

    top_takeoff = _geometry_takeoff(rooms, 1250, 975, 0.5, top_evidence)
    left_takeoff = _geometry_takeoff(rooms, 1250, 975, 0.5, left_evidence)

    failures = []
    if top_takeoff["dimension_source"] != "ocr-width-wall-graph":
        failures.append({"case": "top-only-source", "source": top_takeoff["dimension_source"]})
    if left_takeoff["dimension_source"] != "ocr-depth-wall-graph":
        failures.append({"case": "left-only-source", "source": left_takeoff["dimension_source"]})
    if top_takeoff["shared_wall_length_ft"] <= 0 or left_takeoff["shared_wall_length_ft"] <= 0:
        failures.append({"case": "shared-wall-length", "top": top_takeoff["shared_wall_length_ft"], "left": left_takeoff["shared_wall_length_ft"]})
    return {"cases": 3, "passed": 3 - len(failures), "failed": len(failures), "failures": failures}


def run_dxf_geometry_regression():
    dxf_meta = {
        "total_area": 330,
        "room_area_sum_sqft": 1120,
        "largest_room_area_sqft": 330,
        "bedrooms": 2,
        "bathrooms": 2,
        "kitchens": 1,
        "halls": 1,
        "line_count": 96,
        "geometry_bounds": {"width_units": 84, "height_units": 63},
        "text_dimension_evidence": {"dimension_count": 0, "confidence": 0},
    }
    label_hints = {"named_zone_count": 7, "total_area": 330}
    base_area = _choose_plan_area(dxf_meta, label_hints, {}, 1250)
    evidence = _dxf_dimension_evidence(dxf_meta, base_area)
    takeoff = _geometry_takeoff(_demo_rooms(base_area), base_area, base_area * 0.78, 0.45, evidence)

    failures = []
    if abs(base_area - 1250) > 0.1:
        failures.append({"case": "base-area", "base_area": base_area})
    if takeoff["dimension_source"] != "dxf-envelope-area-calibrated":
        failures.append({"case": "dimension-source", "source": takeoff["dimension_source"]})
    if evidence.get("confidence", 0) < 45:
        failures.append({"case": "dimension-confidence", "confidence": evidence.get("confidence", 0)})
    return {"cases": 3, "passed": 3 - len(failures), "failed": len(failures), "failures": failures}


def run_edge_area_base_area_regression():
    dimension_evidence = {
        "edge_sums_ft": {"top": 49.25, "bottom": 48.33, "left": 25.33, "right": 24.75},
        "edge_binding_axes": 2,
    }
    base_area = _choose_plan_area({}, {"named_zone_count": 0}, dimension_evidence, None)

    failures = []
    if abs(base_area - (49.25 * 25.33)) > 0.2:
        failures.append({"case": "edge-area-fallback", "base_area": base_area})
    return {"cases": 1, "passed": 1 - len(failures), "failed": len(failures), "failures": failures}


def run_anchor_component_regression():
    lines = {
        "vertical": [0, 14, 30, 48, 72, 96, 124],
        "horizontal": [0, 12, 28, 44, 60, 76, 91],
    }
    label = {"type": "bathroom", "cx": 40, "cy": 22, "text": "Bathroom 1 48 sq ft"}
    oversized_component = {"bbox": (6, 6, 92, 72), "pixels": 5700, "source": "component"}
    fitted = _fit_component_to_label(label, oversized_component, lines, 125, 92)

    failures = []
    if not fitted:
        failures.append({"case": "missing-fit"})
    else:
        if fitted.get("bbox") == oversized_component["bbox"]:
            failures.append({"case": "bbox-not-shrunk", "bbox": fitted.get("bbox")})
        if fitted.get("pixels", oversized_component["pixels"]) >= oversized_component["pixels"]:
            failures.append({"case": "pixels-not-reduced", "pixels": fitted.get("pixels")})
        if fitted.get("source") not in {"wall-line-snap", "label-fallback-fit"}:
            failures.append({"case": "unexpected-source", "source": fitted.get("source")})
    return {"cases": 3, "passed": 3 - len(failures), "failed": len(failures), "failures": failures}


def run_edge_bound_dimension_regression():
    rooms = _demo_rooms(2500)
    boxes = [
        _dimension_box(24.0, "top", 22),
        _dimension_box(8.0, "top", 58),
        _dimension_box(14.0, "left", 18),
        _dimension_box(10.0, "left", 50),
    ]
    evidence = _extract_dimension_evidence("24' 8' 14' 10' 2500 SQ FT", 2500, boxes)
    takeoff = _geometry_takeoff(rooms, 2500, 1950, 0.5, evidence)

    failures = []
    if evidence.get("exterior_width_ft") or evidence.get("exterior_depth_ft"):
        failures.append({"case": "edge-spans-should-stay-unbound", "width": evidence.get("exterior_width_ft"), "depth": evidence.get("exterior_depth_ft")})
    if takeoff["dimension_source"] != "ocr-edge-bound-graph":
        failures.append({"case": "dimension-source", "source": takeoff["dimension_source"]})
    if abs(takeoff["external_width_ft"] - 32.0) > 0.2 or abs(takeoff["external_depth_ft"] - 24.0) > 0.2:
        failures.append({"case": "recovered-spans", "width": takeoff["external_width_ft"], "depth": takeoff["external_depth_ft"]})
    if evidence.get("confidence", 0) < 70:
        failures.append({"case": "dimension-confidence", "confidence": evidence.get("confidence", 0)})
    return {"cases": 4, "passed": 4 - len(failures), "failed": len(failures), "failures": failures}


def run_learning_ratio_regression():
    learning = {"internal_wall_length_per_sqft": 0.094}
    takeoff = _geometry_takeoff([], 1250, 975, 0.5, {}, {}, learning)

    failures = []
    if takeoff["internal_wall_source"] != "learning-calibrated-fallback":
        failures.append({"case": "internal-wall-source", "source": takeoff["internal_wall_source"]})
    if abs(takeoff["internal_wall_length_ft"] - 117.5) > 1.0:
        failures.append({"case": "internal-wall-length", "length": takeoff["internal_wall_length_ft"]})
    return {"cases": 2, "passed": 2 - len(failures), "failed": len(failures), "failures": failures}


def run_wall_graph_tightening_regression():
    rooms = [
        {
            "id": "common-zone",
            "type": "living",
            "label": "Common Zone 2",
            "x": 40.0,
            "y": 10.0,
            "width": 40.0,
            "height": 30.0,
            "area_sqft": 240.0,
            "confidence": 0.62,
            "source": "ocr-anchor-1",
            "label_x": 46.0,
            "label_y": 16.0,
        },
        {
            "id": "bed-1",
            "type": "bedroom",
            "label": "Bedroom 1",
            "x": 40.0,
            "y": 10.0,
            "width": 14.0,
            "height": 14.0,
            "area_sqft": 120.0,
            "confidence": 0.84,
            "source": "wall-line-snap",
            "label_x": 46.0,
            "label_y": 16.0,
        },
        {
            "id": "bath-1",
            "type": "bathroom",
            "label": "Bathroom 1",
            "x": 54.0,
            "y": 10.0,
            "width": 12.0,
            "height": 14.0,
            "area_sqft": 48.0,
            "confidence": 0.78,
            "source": "wall-line-snap",
            "label_x": 58.0,
            "label_y": 16.0,
        },
        {
            "id": "kitchen-1",
            "type": "kitchen",
            "label": "Kitchen",
            "x": 66.0,
            "y": 10.0,
            "width": 14.0,
            "height": 14.0,
            "area_sqft": 86.0,
            "confidence": 0.79,
            "source": "wall-line-snap",
            "label_x": 72.0,
            "label_y": 16.0,
        },
        {
            "id": "service-1",
            "type": "service",
            "label": "Service Zone 1",
            "x": 40.0,
            "y": 24.0,
            "width": 14.0,
            "height": 16.0,
            "area_sqft": 60.0,
            "confidence": 0.72,
            "source": "wall-line-snap",
            "label_x": 46.0,
            "label_y": 31.0,
        },
    ]
    refined = _refine_rooms_with_wall_graph(rooms)
    graph = refined["graph"]
    common_zone = next(room for room in refined["rooms"] if room["id"] == "common-zone")
    bedroom = next(room for room in refined["rooms"] if room["id"] == "bed-1")

    failures = []
    if graph.get("tightened_room_count", 0) < 1:
        failures.append({"case": "missing-tightening", "graph": graph})
    if common_zone["width"] >= 40.0 or common_zone["height"] >= 30.0:
        failures.append({"case": "common-zone-not-tightened", "width": common_zone["width"], "height": common_zone["height"]})
    label_distance = ((float(common_zone["label_x"]) - float(bedroom["label_x"])) ** 2 + (float(common_zone["label_y"]) - float(bedroom["label_y"])) ** 2) ** 0.5
    if label_distance < 5.4:
        failures.append({"case": "label-conflict-persists", "distance": round(label_distance, 2)})
    return {"cases": 3, "passed": 3 - len(failures), "failed": len(failures), "failures": failures}


def run_learning_hint_regression():
    base_hints = {
        "bedrooms": 3,
        "bathrooms": 4,
        "kitchens": 1,
        "halls": 3,
        "outdoor_zones": 1,
        "service_zones": 2,
        "stair_zones": 0,
        "primary_room_count": 8,
        "named_zone_count": 14,
        "has_label_hint": True,
    }
    learning = {
        "corrections_applied": 3,
        "room_type_counts": {
            "bedroom": 2,
            "bathroom": 2,
            "kitchen": 1,
            "living": 1,
            "balcony": 1,
            "service": 1,
        },
    }
    enriched = _apply_learning_hints(base_hints, learning)

    failures = []
    if enriched["bathrooms"] != 3:
        failures.append({"case": "bathroom-cap", "bathrooms": enriched["bathrooms"]})
    if enriched["halls"] != 2:
        failures.append({"case": "hall-cap", "halls": enriched["halls"]})
    if enriched["named_zone_count"] != 10:
        failures.append({"case": "named-zone-recount", "named_zone_count": enriched["named_zone_count"]})
    return {"cases": 3, "passed": 3 - len(failures), "failed": len(failures), "failures": failures}


if __name__ == "__main__":
    random_result = run_regression()
    partial_result = run_partial_dimension_regression()
    dxf_result = run_dxf_geometry_regression()
    edge_area_result = run_edge_area_base_area_regression()
    anchor_result = run_anchor_component_regression()
    edge_bound_result = run_edge_bound_dimension_regression()
    learning_result = run_learning_ratio_regression()
    wall_graph_result = run_wall_graph_tightening_regression()
    learning_hint_result = run_learning_hint_regression()
    result = {
        "randomized": random_result,
        "partial_dimension": partial_result,
        "dxf_geometry": dxf_result,
        "edge_area_base": edge_area_result,
        "anchor_component": anchor_result,
        "edge_bound_dimension": edge_bound_result,
        "learning_ratio": learning_result,
        "wall_graph_tightening": wall_graph_result,
        "learning_hints": learning_hint_result,
        "failed": random_result["failed"] + partial_result["failed"] + dxf_result["failed"] + edge_area_result["failed"] + anchor_result["failed"] + edge_bound_result["failed"] + learning_result["failed"] + wall_graph_result["failed"] + learning_hint_result["failed"],
    }
    print(result)
    raise SystemExit(0 if result["failed"] == 0 else 1)
