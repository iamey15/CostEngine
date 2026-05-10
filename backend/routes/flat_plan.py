from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from routes.project import current_user_id
from services.database import Project, session_scope
from services.flat_plan_engine import confirm_layout, detect_layout, estimate_flat, register_demo_plan, register_plan, relabel_edited_zones

router = APIRouter(tags=["flat-wise-estimation"])

PLAN_NOT_FOUND = "Plan not found"


class DetectLayoutRequest(BaseModel):
    project_id: int
    plan_id: str
    reprocess_attempt: int = 0


class ConfirmLayoutRequest(BaseModel):
    project_id: int
    plan_id: str
    summary: dict
    rooms: list | None = None
    user_corrections: int = 0


class RelabelZonesRequest(BaseModel):
    project_id: int
    plan_id: str
    rooms: list
    summary: dict | None = None
    user_corrections: int = 1


class EstimateFlatRequest(BaseModel):
    project_id: int
    plan_id: str
    rate_per_sqft: float = Field(gt=0)
    tier: str = "Standard"


def _get_project(project_id: int, user_id: int):
    with session_scope() as db:
        project = db.get(Project, project_id)
        if not project or project.user_id != user_id:
            raise HTTPException(status_code=404, detail="Project not found")
        return {"id": project.id, "area": float(project.area)}


@router.post("/upload-plan")
async def upload_plan(
    project_id: int = Form(...),
    demo_sample: bool = Form(False),
    file: UploadFile | None = File(default=None),
    authorization: str | None = Header(default=None),
):
    user_id = current_user_id(authorization)
    _get_project(project_id, user_id)
    if demo_sample:
        return register_demo_plan(project_id)
    if not file:
        raise HTTPException(status_code=400, detail="Upload a PNG, JPG, DXF, or PDF file.")
    ext = file.filename.split(".")[-1].lower()
    if ext not in {"png", "jpg", "jpeg", "dxf", "pdf"}:
        raise HTTPException(status_code=400, detail="Supported files: PNG, JPG, JPEG, DXF, PDF.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return register_plan(file.filename, data, project_id)


@router.post("/detect-layout")
def run_layout_detection(payload: DetectLayoutRequest, authorization: str | None = Header(default=None)):
    user_id = current_user_id(authorization)
    project = _get_project(payload.project_id, user_id)
    try:
        return detect_layout(payload.plan_id, project["area"], payload.reprocess_attempt)
    except KeyError as exc:
        if str(exc).strip("'\"") != PLAN_NOT_FOUND:
            raise
        raise HTTPException(status_code=404, detail="Uploaded plan not found")


@router.post("/confirm-layout")
def confirm_detected_layout(payload: ConfirmLayoutRequest, authorization: str | None = Header(default=None)):
    user_id = current_user_id(authorization)
    _get_project(payload.project_id, user_id)
    try:
        return confirm_layout(payload.plan_id, payload.summary, payload.rooms, payload.user_corrections)
    except KeyError as exc:
        if str(exc).strip("'\"") != PLAN_NOT_FOUND:
            raise
        raise HTTPException(status_code=404, detail="Uploaded plan not found")


@router.post("/relabel-zones")
def relabel_corrected_zones(payload: RelabelZonesRequest, authorization: str | None = Header(default=None)):
    user_id = current_user_id(authorization)
    _get_project(payload.project_id, user_id)
    try:
        return relabel_edited_zones(payload.plan_id, payload.rooms, payload.summary, payload.user_corrections)
    except KeyError as exc:
        if str(exc).strip("'\"") != PLAN_NOT_FOUND:
            raise
        raise HTTPException(status_code=404, detail="Uploaded plan not found")


@router.post("/estimate-flat")
def estimate_flat_layout(payload: EstimateFlatRequest, authorization: str | None = Header(default=None)):
    user_id = current_user_id(authorization)
    _get_project(payload.project_id, user_id)
    try:
        return estimate_flat(payload.plan_id, payload.rate_per_sqft, payload.tier)
    except KeyError as exc:
        if str(exc).strip("'\"") != PLAN_NOT_FOUND:
            raise
        raise HTTPException(status_code=404, detail="Uploaded plan not found")
