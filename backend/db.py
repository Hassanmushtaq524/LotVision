from sqlalchemy import create_engine
from fastapi import HTTPException, Depends
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# SQLite database file path
DB_PATH = os.getenv("DB_PATH", "sqlite:///./local_parking_lot.db")

# Create SQLite engine
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})  # Required for SQLite to work with FastAPI

# Create session and base class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get DB session
async def get_db():
    db = SessionLocal()
    try: 
        yield db
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        db.close()
        
db_dependency = Depends(get_db)
