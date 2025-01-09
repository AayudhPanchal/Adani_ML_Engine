from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import traceback
from GradientBoost_2 import predict_usage_for_date, predict_multiple_dates

app = Flask(__name__)
CORS(app)

# Load historical data once when server starts
try:
    historical_data = pd.read_csv(
        r'E:\Programs\Adani_Thinkbiz\ML_Models\Datasets\Merged_Datasets\train_preprocessed.csv',
        index_col='Dates',
        parse_dates=True
    )
except Exception as e:
    print(f"Error loading historical data: {str(e)}")
    historical_data = None

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
    return jsonify({
        'status': 'healthy',
        'model_loaded': historical_data is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
