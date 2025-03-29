from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from starlette.requests import Request
from pydantic import BaseModel
from db import db_dependency
from models import RegisteredCars 

router = APIRouter()

# GET ALL REGISTERED CARS
@router.get("/{lot_id}")
def get_registered_cars(request: Request, lot_id: str, db: Session = db_dependency):
    try:
        # Fetch all registered cars from the database
        cars = db.query(RegisteredCars).filter(RegisteredCars.lot_id == lot_id).all()
        
        if not cars:
            return JSONResponse(status_code=404, content={"detail": "No registered cars found"})
        
        # Convert the list of car objects into dictionaries
        cars_list = [{
            "plate_num": car.plate_num,
            "owner_name": car.owner_name,
            "lot_id": car.lot_id
        } for car in cars]
        
        return {"status": "success", "registered_cars": cars_list}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"{e}"})



class AddCarBody(BaseModel):
    lot_id: str
    plate_num: str
    owner_name: str
    email: str

# ADD REGISTERED CAR
@router.post("/")
def add_registered_cars(request: Request, data: AddCarBody, db: Session = db_dependency):
    new_car = RegisteredCars(lot_id=data.lot_id, 
                             owner_name=data.owner_name, 
                             plate_num=data.plate_num, 
                             email=data.email)
    db.add(new_car)  
    db.commit()     
    db.refresh(new_car) 
    return {"message": "Car added successfully", "car": new_car}
