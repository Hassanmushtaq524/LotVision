from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import JSONResponse 
from starlette.middleware.sessions import SessionMiddleware 
from starlette.requests import Request
import uvicorn 
from dotenv import load_dotenv 
from routers.registered_cars import router as registered_cars_router
from routers.lots import router as lots_router
from routers.current_cars import router as current_cars_router
# from db import Base, engine, db_dependency
from sqlalchemy.orm import Session
from typing import Annotated
import os
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()


FRONTEND_URL = os.getenv("FRONTEND_URL")
SECRET_KEY = os.getenv("SECRET_KEY")


if not FRONTEND_URL:
    logging.warning("FRONTEND_URL is not set.")
if not SECRET_KEY:
    logging.warning("SECRET_KEY is not set.")


app = FastAPI() 
origins = [ 
	FRONTEND_URL
]
 

app.add_middleware( 
	CORSMiddleware, 
	allow_origins=origins, 
	allow_credentials=True, 
	allow_methods=["*"], 
	allow_headers=["*"], 
) 

app.add_middleware(SessionMiddleware, 
                   secret_key=SECRET_KEY
) 


# registered cars route
app.include_router(router=registered_cars_router, prefix="/registered_cars")
# lots route
app.include_router(router=lots_router, prefix="/lots")
# current cars route
app.include_router(router=current_cars_router, prefix="/current_cars")


if __name__ == "__main__": 
	uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")