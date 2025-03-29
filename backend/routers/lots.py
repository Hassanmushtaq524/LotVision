from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from starlette.requests import Request
from pydantic import BaseModel
from db import db_dependency
from models import Lots 

router = APIRouter()


class AddLotBody(BaseModel):
    lot_id: str
    capacity: int

# GET ALL REGISTERED CARS
@router.post("/")
def add_lots(request: Request, data: AddLotBody, db: Session = db_dependency):
    try:
        new_lot = Lots(lot_id=data.lot_id, capacity=data.capacity)
        db.add(new_lot)
        db.commit()
        db.refresh(new_lot)
        return {"status": "success"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"{e}"})

