import csv
import random
import requests

# API endpoint
API_URL = "http://localhost:8000/current_cars/"

# Available parking lots
AUTHORIZED_LOTS = ["A", "B", "C", "D"]

# Read CSV file
csv_file_path = "parking_data.csv"  # Update if needed

def assign_cars_to_lots():
    try:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                car_plate_num = row.get("plate_number")
                registered_lot = row.get("lot_id")

                # Sometimes assign the car to an unauthorized lot
                if random.random() < 0.1:  # 10% chance of wrong lot
                    available_lots = [lot for lot in AUTHORIZED_LOTS if lot != registered_lot]
                    lot_id = random.choice(available_lots)  # Wrong lot
                else:
                    lot_id = registered_lot  # Correct lot

                # Prepare car entry data
                car_data = {
                    "car_plate_num": car_plate_num,
                    "lot_id": lot_id,
                }

                # Send data to API
                response = requests.post(API_URL, json=car_data)

                # Print response
                if response.status_code == 200:
                    print(f"✅ Car {car_plate_num} entered Lot {lot_id} (Registered: {registered_lot})")
                else:
                    print(f"❌ Failed to enter {car_plate_num} in Lot {lot_id}: {response.text}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    assign_cars_to_lots()
