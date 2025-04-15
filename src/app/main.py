from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
from pathlib import Path
import os
from ..models.inference import LoanPredictor

app = FastAPI(title="Loan Prediction API")

# Set up templates and static files
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Initialize predictor
predictor = LoanPredictor()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    # Read the uploaded file
    df = pd.read_csv(file.file)
    
    # Make predictions
    predictions = predictor.predict(df)
    
    # Convert predictions to HTML table
    predictions_html = predictions.to_html(
        classes="table table-striped table-hover",
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
    
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "predictions": predictions_html,
            "num_predictions": len(predictions)
        }
    )

@app.get("/predict-single", response_class=HTMLResponse)
async def predict_single_form(request: Request):
    return templates.TemplateResponse("single_prediction.html", {"request": request})

@app.post("/predict-single", response_class=HTMLResponse)
async def predict_single(
    request: Request,
    account_id: str = Form(...),
    file: UploadFile = File(...)
):
    # Read the uploaded file
    df = pd.read_csv(file.file)
    
    # Filter for the specific account
    account_data = df[df['Account ID'] == account_id]
    
    if len(account_data) == 0:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error": f"Account {account_id} not found in the data"
            }
        )
    
    # Make prediction
    prediction = predictor.predict_single_account(account_data)
    
    return templates.TemplateResponse(
        "single_result.html",
        {
            "request": request,
            "account_id": account_id,
            "prediction": prediction
        }
    ) 