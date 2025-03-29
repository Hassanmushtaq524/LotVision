from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from starlette.requests import Request
from pydantic import BaseModel
from db import db_dependency
from models import CurrentCars, RegisteredCars

router = APIRouter()


class AddCurrentCarsBody(BaseModel):
    car_plate_num: str
    lot_id: str

# ADD A CAR AS IT ENTERS, IF DETECTED AGAIN, ITS LEAVING AND DELETE
@router.post("/")
def add_or_remove_car(request: Request, data: AddCurrentCarsBody, db: Session = db_dependency):
    try:
        # Check if the car is already in the lot
        existing_car = db.query(CurrentCars).filter(CurrentCars.car_plate_num == data.car_plate_num).first()

        if existing_car:
            # Car is leaving, remove it from the table
            db.delete(existing_car)
            db.commit()
            return {"status": "success", "message": f"Car {data.car_plate_num} has left the lot.", "lot_id": data.lot_id}
        else:
            # Car is entering, add it to the table
            new_car = CurrentCars(lot_id=data.lot_id, car_plate_num=data.car_plate_num)
            db.add(new_car)
            db.commit()
            db.refresh(new_car)
            return {"status": "success", "message": f"Car {new_car.car_plate_num} entered the lot.", "lot_id": new_car.lot_id}

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
                "enter_time": current_car.enter_time,
            }
            for current_car, registered_car in cars_in_lot
        ]

        return {"status": "success", "lot_id": lot_id, "cars": cars_data}

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

