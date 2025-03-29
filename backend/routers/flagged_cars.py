from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from starlette.requests import Request
from pydantic import BaseModel
from db import db_dependency
from models import FlaggedCars

router = APIRouter()

# GET LOGS OF CARS
# USE GEMINI HERE
@router.get("/")
def get_cars(request: Request, db: Session = db_dependency):
    try:
        # Fetch all cars currently in the given lot along with their registered lot info
        flagged_cars = (
            db.query(FlaggedCars).all()
        )

        # Format response
        cars_data = [
            {
                "car_plate_num": car.car_plate_num,
                "detected_lot_id": car.detected_lot_id,
                "registered_lot_id": car.registered_lot_id,
                "flag_reason": car.flag_reason,
                "flag_time": car.flag_time,
            }
            for car in flagged_cars
        ]
        return {"status": "success", "cars": cars_data}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
