import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from ..models import InventoryLog, Product
from datetime import datetime, timedelta

def predict_shortage(db: Session, product_id: int):
    """
    Predicts when stock will run out using a simple Moving Average for now.
    Can be upgraded to XGBoost/LSTM.
    """
    # Fetch product
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        return None
    
    # Fetch logs
    logs = db.query(InventoryLog).filter(InventoryLog.product_id == product_id).order_by(InventoryLog.date).all()
    
    if not logs or len(logs) < 3:
        # Not enough data, return heuristic
        # Assume standard usage if no data
        predicted_daily_usage = 5.0 # default dummy
        run_out_days = product.current_stock / predicted_daily_usage if predicted_daily_usage > 0 else 999
        return {
            "product_id": product_id,
            "predicted_daily_usage": predicted_daily_usage,
            "run_out_in_days": round(run_out_days, 1),
            "recommended_restock": 50.0,
            "confidence_score": 0.5
        }

    # Prepare DataFrame
    data = [{"date": log.date, "sold": log.sold_qty} for log in logs]
    df = pd.DataFrame(data)
    
    # Simple Moving Average (7 days)
    # In a real scenario, we would resample to daily if logs are irregular
    # For now, assume one log per day or take mean
    
    avg_sold = df["sold"].mean() # Simple mean
    # Or last 7 entries mean
    if len(df) >= 7:
        avg_sold = df["sold"].tail(7).mean()
        
    predicted_daily_usage = avg_sold if avg_sold > 0 else 1.0
    
    run_out_days = product.current_stock / predicted_daily_usage
    
    # Recommended restock: Target 30 days stock
    target_stock = predicted_daily_usage * 30
    restock_qty = max(0, target_stock - product.current_stock)
    
    return {
        "product_id": product_id,
        "predicted_daily_usage": round(predicted_daily_usage, 2),
        "run_out_in_days": round(run_out_days, 1),
        "recommended_restock": round(restock_qty, 0),
        "confidence_score": 0.85
    }
