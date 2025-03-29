from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from starlette.requests import Request
from pydantic import BaseModel
from db import db_dependency
from models import FlaggedCars
from google import genai
import os
import json
import time

# Maximum number of retries to avoid infinite loop
MAX_RETRIES = 5
# Delay between retries (in seconds)
RETRY_DELAY = 2

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

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


        # USE GEMINI API TO FILTER OUT SUSPICIOUS VEHICLES BASED ON TIME-STAMPS
        # Prepare the prompt for Gemini based on the car data
        prompt = """
            Analyze the following vehicles and flag any suspicious vehicles based on their timestamps, number of entries, and flag reasons.
            Return only the license plate number and a note describing why it is more suspicious than other entries. Sort based on how suspicious the vehicle is. Output only 5 entries.\n 
            Return in the format [{ car_plate_num, note }] for multiple cars. Do not output anything else, and do not output the newline character anywhere.\n\n
        """
        for car in cars_data:
            prompt += f"Car Plate Number: {car['car_plate_num']}, Flag Time: {car['flag_time']}, Flag Reason: {car['flag_reason']}\n"

        # Call Gemini API to analyze the data (send the prompt to Gemini)
        gemini_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        gemini_response_text = gemini_response.text
                
        # Attempt to parse the Gemini response text as JSON
        suspicious_data = json.loads(gemini_response_text)
        # retries = 0
        # suspicious_data = None
        # while retries < MAX_RETRIES:
        #     try:
               
        #         break 
        #     except json.JSONDecodeError as e:
        #         retries += 1
        #         if retries >= MAX_RETRIES:
        #             raise HTTPException(status_code=500, detail="Failed to parse Gemini response after multiple attempts")
        #         # Wait for a brief moment before retrying
        #         time.sleep(RETRY_DELAY)


        return {"status": "success", "cars": cars_data, "suspicious_cars": suspicious_data}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
