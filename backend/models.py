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

