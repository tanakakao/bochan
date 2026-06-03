"""Acquisition registry endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from bochan.api import available_acqf_names

from ..schemas import AcquisitionNamesResponse

router = APIRouter(prefix="/acquisitions", tags=["acquisitions"])


@router.get("/names", response_model=AcquisitionNamesResponse)
def list_acquisition_names() -> AcquisitionNamesResponse:
    return AcquisitionNamesResponse(names=available_acqf_names())
