from db import engine, Base
from models import RegisteredCars, Lots, CurrentCars

# Create all tables in the SQLite database
Base.metadata.create_all(bind=engine)

print("SQLite Database and tables created successfully!")
