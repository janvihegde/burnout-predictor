# backend/main.py (Simplified - No Database)
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
with open("burnout_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define Input
class StudentData(BaseModel):
    Branch: str
    Year: str
    Daily_Study_Hours: float
    Sleep_Hours: float
    Weekly_Assignments: int
    Backlogs: int
    Screen_Time_Hours: float
    Stress_Level_1_10: int
    Career_Clarity_1_10: int
    Interested_in_Core_Job: str

@app.post("/predict")
def predict(data: StudentData):
    # Convert input
    input_data = data.dict()
    df_input = pd.DataFrame([input_data])
    
    # Predict
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]
    
    return {
        "burnout_risk": int(prediction),
        "risk_probability": round(float(probability) * 100, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)