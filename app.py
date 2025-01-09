from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import traceback
from GradientBoost_2 import predict_usage_for_date, predict_multiple_dates
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load historical data once when server starts
try:
    # Use relative path
    data_path = os.path.join(current_dir, 'train_preprocessed.csv')
    logger.info(f"Loading data from: {data_path}")
    
    historical_data = pd.read_csv(
        data_path,
        index_col='Dates',
        parse_dates=True
    )
    logger.info("Historical data loaded successfully")
except Exception as e:
    logger.error(f"Error loading historical data: {str(e)}")
    historical_data = None

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'service running',
        'endpoints': {
            '/predict/single': 'POST - Predict for a single date',
            '/predict/range': 'POST - Predict for a date range',
            '/health': 'GET - Health check'
        }
    })

@app.route('/predict/single', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        
        if not data or 'date' not in data:
            return jsonify({'error': 'Date is required'}), 400
        
        target_date = data['date']
        
        # Validate date format
        try:
            datetime.strptime(target_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        if historical_data is None:
            return jsonify({'error': 'Historical data not loaded'}), 500
        
        prediction = predict_usage_for_date(target_date, historical_data)
        
        return jsonify({
            'date': target_date,
            'predicted_usage': float(prediction)
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/predict/range', methods=['POST'])
def predict_range():
    try:
        data = request.get_json()
        
        if not data or 'start_date' not in data or 'end_date' not in data:
            return jsonify({'error': 'Start date and end date are required'}), 400
        
        start_date = data['start_date']
        end_date = data['end_date']
        
        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        if historical_data is None:
            return jsonify({'error': 'Historical data not loaded'}), 500
        
        predictions_df = predict_multiple_dates(start_date, end_date, historical_data)
        
        # Convert predictions to list of dictionaries
        predictions = predictions_df.reset_index().to_dict(orient='records')
        predictions = [{
            'date': row['Date'].strftime('%Y-%m-%d'),
            'predicted_usage': float(row['Predicted_Usage'])
        } for row in predictions]
        
        return jsonify({
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        'status': 'healthy' if historical_data is not None else 'degraded',
        'model_loaded': historical_data is not None,
        'debug_info': {
            'current_dir': current_dir,
            'python_path': os.getenv('PYTHONPATH', 'Not set'),
            'working_dir': os.getcwd()
        }
    }
    return jsonify(status)

def initialize_app():
    """Initialize the Flask app and verify all dependencies"""
    try:
        # Verify model file exists
        model_path = os.path.join(current_dir, 'energy_prediction_model.pkl')
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        # Verify data directory exists
        data_dir = os.path.join(current_dir, 'data')
        if not os.path.exists(data_dir):
            logger.info(f"Creating data directory at: {data_dir}")
            os.makedirs(data_dir)
        
        logger.info("App initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing app: {str(e)}")
        return False

if __name__ == '__main__':
    if initialize_app():
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to initialize app. Please check the logs.")
