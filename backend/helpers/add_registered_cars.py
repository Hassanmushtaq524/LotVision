import csv
import requests

# Define the API endpoint
API_URL = "http://localhost:8000/registered_cars/"

# Read CSV file
csv_file_path = "parking_data.csv"  # Update this if needed

def add_registered_cars():
    try:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Extract necessary fields
                car_data = {
                    "plate_num": row.get("plate_number"),
                    "owner_name": row.get("owner_name"),
                    "email": row.get("email"),
                    "lot_id": row.get("lot_id"),
                }

                # Send data to API
                response = requests.post(API_URL, json=car_data)

                # Print response
                if response.status_code == 201:
                    print(f"Successfully added {car_data['plate_num']}")
                else:
                    print(f"Failed to add {car_data['plate_num']}: {response.text}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    add_registered_cars()
