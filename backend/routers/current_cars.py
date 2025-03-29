from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from starlette.requests import Request
from pydantic import BaseModel
from db import db_dependency
from models import CurrentCars, RegisteredCars, FlaggedCars
from datetime import datetime

router = APIRouter()


class AddCurrentCarsBody(BaseModel):
    car_plate_num: str
    lot_id: str


@router.post("/")
def add_or_remove_car(request: Request, data: AddCurrentCarsBody, db: Session = db_dependency):
    try:
        # Check if the car is already in the lot (leaving case)
        existing_car = db.query(CurrentCars).filter(CurrentCars.car_plate_num == data.car_plate_num).first()

        if existing_car:
            # Car is leaving, remove it from the table
            db.delete(existing_car)
            db.commit()
            return {
                "status": "success",
                "message": f"Car {data.car_plate_num} has left the lot.",
                "lot_id": data.lot_id
            }

        # Check if the car is registered
        registered_car = db.query(RegisteredCars).filter(RegisteredCars.plate_num == data.car_plate_num).first()

        # Car is entering, create new CurrentCars entry
        new_car = CurrentCars(lot_id=data.lot_id, car_plate_num=data.car_plate_num)
        db.add(new_car)

        flagged = False
        flag_reason = None

        if not registered_car:
            # Unregistered car detected
            flag_reason = "Unregistered Car"
            flagged = True
        elif registered_car.lot_id != data.lot_id:
            # Registered car in the wrong lot
            flag_reason = "Unauthorized Car"
            flagged = True

        # If flagged, add to the flagged_cars table
        if flagged:
            flagged_car = FlaggedCars(
                car_plate_num=data.car_plate_num,
                detected_lot_id=data.lot_id,
                registered_lot_id=registered_car.lot_id if registered_car else None,
                flag_reason=flag_reason,
                flag_time=datetime.utcnow()
            )
            db.add(flagged_car)

        db.commit()
        db.refresh(new_car)

        return {
            "status": "success",
            "message": f"Car {new_car.car_plate_num} entered the lot.",
            "lot_id": new_car.lot_id,
            "flagged": flagged,
            "flag_reason": flag_reason
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})




# GET ALL CURRENT CARS IN A LOT
@router.get("/{lot_id}")
def get_cars(request: Request, lot_id: str, db: Session = db_dependency):
    try:
        # Fetch all cars currently in the given lot along with their registered lot info
        cars_in_lot = (
            db.query(CurrentCars, RegisteredCars)
            .join(RegisteredCars, CurrentCars.car_plate_num == RegisteredCars.plate_num)
            .filter(CurrentCars.lot_id == lot_id)
            .all()
        )

        if not cars_in_lot:
            return JSONResponse(status_code=404, content={"detail": "No cars found in this lot."})

        # Format response
        cars_data = [
            {
                "car_plate_num": current_car.car_plate_num,
                "registered_lot_id": registered_car.lot_id,
                "registered_email": registered_car.email,
                "owner_name": registered_car.owner_name,
                "enter_time": current_car.enter_time,
            }
            for current_car, registered_car in cars_in_lot
        ]

        return {"status": "success", "cars": cars_data}


    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

