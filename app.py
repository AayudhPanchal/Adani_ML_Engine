from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import traceback
from GradientBoost_2 import predict_usage_for_date, predict_multiple_dates
from powerinference import PowerPredictor 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load historical data once when server starts
try:
    historical_data = pd.read_csv('./train_preprocessed.csv',
        index_col='Dates',
        parse_dates=True
    )
except Exception as e:
    print(f"Error loading historical data: {str(e)}")
    historical_data = None

# Create an object of PowerPredictor class
power_predictor = PowerPredictor()

class SinglePredictionRequest(BaseModel):
    date: str

class RangePredictionRequest(BaseModel):
    start_date: str
    end_date: str

@app.post("/predict/single")
async def predict_single(request: SinglePredictionRequest):
    try:
        target_date = request.date
        
        # Validate date format
        try:
            datetime.strptime(target_date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        if historical_data is None:
            raise HTTPException(status_code=500, detail="Historical data not loaded")
        
        prediction = predict_usage_for_date(target_date, historical_data)
        
        return {
            'date': target_date,
            'predicted_usage': float(prediction)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            'error': 'Prediction failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        })

@app.post("/predict/range")
async def predict_range(request: RangePredictionRequest):
    try:
        start_date = request.start_date
        end_date = request.end_date
        
        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        if historical_data is None:
            raise HTTPException(status_code=500, detail="Historical data not loaded")
        
        predictions_df = predict_multiple_dates(start_date, end_date, historical_data)
        
        # Convert predictions to list of dictionaries
        predictions = predictions_df.reset_index().to_dict(orient='records')
        predictions = [{
            'date': row['Date'].strftime('%Y-%m-%d'),
            'predicted_usage': float(row['Predicted_Usage'])
        } for row in predictions]
        
        return {
            'predictions': predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            'error': 'Prediction failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        })

@app.post("/predict/power")
async def power_predict(request: RangePredictionRequest):
    try:
        start_date = request.start_date
        end_date = request.end_date
        
        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Call the infer method from PowerPredictor class
        predictions = power_predictor.infer(start_date, end_date)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            'error': 'Prediction failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        })

@app.get("/health")
async def health_check():
    return {'status': 'healthy'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)