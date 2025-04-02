from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd
from fastapi.responses import JSONResponse
from tabulate import tabulate


with open("model.pkl","rb") as f:
    model = pickle.load(f)
with open("x_train.pkl", "rb") as r:
    x_train = pickle.load(r)



app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


class PredictionInput(BaseModel):
    name: str
    company: str
    year: int
    kms_driven: int
    fuel_type: str


@app.get("/")
def home():

    x_train_table = tabulate(x_train, headers='keys', tablefmt='grid')

    # Return x_train table as plain text
    return {"x_train_table": x_train_table}

@app.post("/predict")
def predict(data: PredictionInput):
    try:


        input_df = pd.DataFrame([{
            "name":data.name,
            "company":data.company,
            "year": data.year,
            "kms_driven": data.kms_driven,
            "fuel_type": data.fuel_type
        }])

        transformed_input = model.named_steps["step1"].transform(input_df)
        prediction = model.named_steps["step2"].predict(transformed_input)[0]



        return {"predicted_price": round(prediction, 2)}
        
    except Exception as e:
        return {"error": str(e)}    