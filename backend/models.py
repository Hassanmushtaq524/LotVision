from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from db import Base
from datetime import datetime

# RegisteredCars table
class RegisteredCars(Base):
    __tablename__ = "registered_cars"
    _id = Column(Integer, primary_key=True, autoincrement=True)
    plate_num = Column(String, unique=True, nullable=False) 
    owner_name = Column(String, nullable=False)
    lot_id = Column(String, nullable=False)  
    email = Column(String, nullable=False)
    

# Lots table
class Lots(Base):
    __tablename__ = "lots"
    _id = Column(Integer, primary_key=True, index=True) 
    lot_id = Column(String, unique=True, nullable=False)
    capacity = Column(Integer, nullable=False)
    occupied = Column(Integer, default=0)


# CurrentCars table
class CurrentCars(Base):
    __tablename__ = "current_cars"
    _id = Column(Integer, primary_key=True, index=True)
    lot_id = Column(Integer, ForeignKey("lots.lot_id"), nullable=False)
    car_plate_num = Column(String, ForeignKey("registered_cars.plate_num"), nullable=False)  
    enter_time = Column(DateTime, default=datetime.utcnow)


# FlaggedCars table
class FlaggedCars(Base):
    __tablename__ = "flagged_cars"
    _id = Column(Integer, primary_key=True, autoincrement=True)
    car_plate_num = Column(String, ForeignKey("registered_cars.plate_num"), nullable=False)
    detected_lot_id = Column(String, ForeignKey("lots.lot_id"), nullable=False)
    registered_lot_id = Column(String, ForeignKey("lots.lot_id"), nullable=True)
    flag_reason = Column(String, nullable=False, default="Unauthorized Parking")
    flag_time = Column(DateTime, default=datetime.utcnow)
