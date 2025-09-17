import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from datetime import datetime
import json
import re
from typing import Dict, List, Optional
import google.generativeai as genai
import os
from dataclasses import dataclass
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InventoryPrediction:
    product_id: str
    current_stock: int
    predicted_shortage_days: int
    confidence_score: float
    recommended_reorder_quantity: int
    severity: str

@dataclass
class StructuredOrder:
    products: List[Dict]
    supplier_category: str
    delivery_date: Optional[str]
    budget_range: Optional[str]
    special_requirements: List[str]
    urgency: str
    confidence_score: float

class InventoryMLPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = [
            'current_stock', 'avg_daily_sales', 'days_since_last_order',
            'seasonal_factor', 'promotion_active', 'day_of_week',
            'month', 'category_encoded', 'supplier_reliability_score'
        ]

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
        data['month'] = pd.to_datetime(data['date']).dt.month
        data['avg_daily_sales'] = data.groupby('product_id')['daily_sales'].rolling(7).mean().reset_index(0, drop=True)
        data['avg_daily_sales'] = data['avg_daily_sales'].fillna(data['daily_sales'])
        data['days_since_last_order'] = data.groupby('product_id')['date'].diff().dt.days.fillna(0)
        data['seasonal_factor'] = np.sin(2 * np.pi * data['month'] / 12) + 1
        if 'category' in data.columns:
            if 'category' not in self.label_encoders:
                self.label_encoders['category'] = LabelEncoder()
                data['category_encoded'] = self.label_encoders['category'].fit_transform(data['category'])
            else:
                data['category_encoded'] = self.label_encoders['category'].transform(data['category'])
        return data

    def train(self, historical_data: pd.DataFrame):
        logger.info("Training inventory prediction model...")
        data = self.prepare_features(historical_data.copy())
        data['days_to_shortage'] = np.where(
            data['daily_sales'] > 0,
            np.maximum(0, (data['current_stock'] - data['critical_threshold']) / data['daily_sales']),
            30
        )
        X = data[self.feature_columns].fillna(0)
        y = data['days_to_shortage']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.info(f"Model trained successfully. MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        return {
            'mae': mae,
            'rmse': rmse,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }

    def predict_shortages(self, current_inventory: pd.DataFrame) -> List[InventoryPrediction]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        data = self.prepare_features(current_inventory.copy())
        X = data[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        results = []
        for idx, row in data.iterrows():
            shortage_days = max(0, int(predictions[idx]))
            confidence = min(1.0, 1.0 / (1.0 + abs(shortage_days - np.mean(predictions)) / np.std(predictions)))
            if shortage_days <= 3:
                severity = 'critical'
            elif shortage_days <= 7:
                severity = 'high'
            elif shortage_days <= 14:
                severity = 'medium'
            else:
                severity = 'low'
            reorder_qty = max(
                int(row['avg_daily_sales'] * 14),
                row.get('minimum_order_quantity', 10)
            )
            results.append(InventoryPrediction(
                product_id=row['product_id'],
                current_stock=row['current_stock'],
                predicted_shortage_days=shortage_days,
                confidence_score=confidence,
                recommended_reorder_quantity=reorder_qty,
                severity=severity
            ))
        return sorted(results, key=lambda x: x.predicted_shortage_days)

    def save_model(self, filepath: str):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }, filepath)

    def load_model(self, filepath: str):
        loaded = joblib.load(filepath)
        self.model = loaded['model']
        self.scaler = loaded['scaler']
        self.label_encoders = loaded['label_encoders']

class NLPOrderProcessor:
    def __init__(self, gemini_api_key: str = None):
        api_key_to_use = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if api_key_to_use:
            genai.configure(api_key=api_key_to_use)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None
            logger.warning("Gemini API Key not found. NLP processing will rely on fallback methods.")
            
        self.product_categories = [
            'dairy', 'meat', 'vegetables', 'fruits', 'bakery', 'beverages',
            'frozen', 'canned_goods', 'snacks', 'household', 'personal_care'
        ]
        self.urgency_keywords = {
            'urgent': ['urgent', 'asap', 'immediately', 'emergency', 'rush'],
            'high': ['soon', 'quickly', 'fast', 'priority'],
            'medium': ['normal', 'regular', 'standard'],
            'low': ['whenever', 'no rush', 'flexible', 'when convenient']
        }

    def extract_products_regex(self, query: str) -> List[Dict]:
        products = []
        quantity_pattern = r'(\d+)\s*(kg|kilos|pounds|lbs|bottles|cans|boxes|units|pieces|packs)?\s*([a-zA-Z\s]+)'
        matches = re.findall(quantity_pattern, query.lower())
        for match in matches:
            quantity = int(match[0])
            unit = match[1] if match[1] else 'units'
            product = match[2].strip()
            products.append({'name': product, 'quantity': quantity, 'unit': unit})
        return products

    def classify_urgency(self, query: str) -> str:
        query_lower = query.lower()
        for urgency_level, keywords in self.urgency_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return urgency_level
        return 'medium'

    def extract_date_requirements(self, query: str) -> Optional[str]:
        date_patterns = [
            r'by (\w+)', r'within (\d+) days?', r'on (\w+ \d+)', r'(\d{1,2}/\d{1,2}/\d{4})'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1)
        return None

    def extract_budget_info(self, query: str) -> Optional[str]:
        budget_patterns = [
            r'budget:?\s*\$?(\d+(?:,\d+)?(?:\.\d{2})?)',
            r'up to \$?(\d+(?:,\d+)?(?:\.\d{2})?)',
            r'maximum \$?(\d+(?:,\d+)?(?:\.\d{2})?)',
            r'under \$?(\d+(?:,\d+)?(?:\.\d{2})?)'
        ]
        for pattern in budget_patterns:
            match = re.search(pattern, query.lower())
            if match:
                return f"${match.group(1)}"
        return None

    def process_with_gemini(self, query: str) -> Optional[Dict]:
        if not self.model:
            return None
        system_prompt = f"""
        You are an AI assistant for a B2B grocery procurement system. Convert the following informal request into a structured JSON object.
        Extract the following:
        1. Products with quantities and units.
        2. A single, most relevant supplier category from: {', '.join(self.product_categories)}.
        3. Delivery requirements as a simple string (e.g., "tomorrow", "within 3 days").
        4. Budget information.
        5. Any special requirements (e.g., "organic", "fresh", "bulk_order").
        6. An urgency level from: low, medium, high, urgent.
        Return ONLY the raw JSON object, without any markdown formatting like ```json.
        """
        full_prompt = f"{system_prompt}\n\nUser Request: \"{query}\""
        try:
            response = self.model.generate_content(full_prompt)
            cleaned_response = re.sub(r'```json\n|```', '', response.text).strip()
            return json.loads(cleaned_response)
        except Exception as e:
            logger.warning(f"Gemini processing failed: {e}. Using fallback methods.")
            return None

    def process_query(self, query: str) -> StructuredOrder:
        logger.info(f"Processing query: {query[:100]}...")
        gemini_result = self.process_with_gemini(query)
        if gemini_result:
            confidence = 0.9
            products = gemini_result.get('products', [])
            supplier_category = gemini_result.get('supplier_category', 'general')
            delivery_date = gemini_result.get('delivery_date')
            budget_range = gemini_result.get('budget_range')
            special_requirements = gemini_result.get('special_requirements', [])
            urgency = gemini_result.get('urgency', 'medium')
        else:
            confidence = 0.7
            products = self.extract_products_regex(query)
            supplier_category = 'general'
            for category in self.product_categories:
                if category in query.lower():
                    supplier_category = category
                    break
            delivery_date = self.extract_date_requirements(query)
            budget_range = self.extract_budget_info(query)
            urgency = self.classify_urgency(query)
            special_requirements = []
            if 'organic' in query.lower(): special_requirements.append('organic')
            if 'fresh' in query.lower(): special_requirements.append('fresh')
            if 'bulk' in query.lower(): special_requirements.append('bulk_order')
        return StructuredOrder(
            products=products,
            supplier_category=supplier_category,
            delivery_date=delivery_date,
            budget_range=budget_range,
            special_requirements=special_requirements,
            urgency=urgency,
            confidence_score=confidence
        )

class GrocerSyncAIEngine:
    def __init__(self, gemini_api_key: str = None):
        self.inventory_predictor = InventoryMLPredictor()
        self.nlp_processor = NLPOrderProcessor(gemini_api_key)
        self.db_connection = None

    def connect_database(self, db_path: str):
        self.db_connection = sqlite3.connect(db_path)

    def get_shortage_alerts(self) -> List[InventoryPrediction]:
        if not self.db_connection:
            raise ValueError("Database not connected")
        query = """
        SELECT product_id, current_stock, daily_sales, category,
               critical_threshold, minimum_order_quantity,
               supplier_reliability_score, promotion_active
        FROM inventory_view
        """
        current_inventory = pd.read_sql_query(query, self.db_connection)
        current_inventory['date'] = datetime.now().strftime('%Y-%m-%d')
        return self.inventory_predictor.predict_shortages(current_inventory)

    def process_procurement_request(self, informal_query: str) -> StructuredOrder:
        return self.nlp_processor.process_query(informal_query)

    def train_prediction_model(self, historical_data_path: str):
        historical_data = pd.read_csv(historical_data_path)
        return self.inventory_predictor.train(historical_data)

    def generate_dashboard_data(self) -> Dict:
        shortage_predictions = self.get_shortage_alerts()
        severity_counts = {}
        for pred in shortage_predictions:
            severity_counts[pred.severity] = severity_counts.get(pred.severity, 0) + 1
        critical_items = [pred for pred in shortage_predictions if pred.severity == 'critical'][:10]
        return {
            'total_products_tracked': len(shortage_predictions),
            'severity_breakdown': severity_counts,
            'critical_items': [{
                'product_id': item.product_id,
                'current_stock': item.current_stock,
                'shortage_days': item.predicted_shortage_days,
                'recommended_quantity': item.recommended_reorder_quantity
            } for item in critical_items],
            'predictions_last_updated': datetime.now().isoformat()
        }



def test_nlp_processing():
    print("\nTesting NLP Processing...")
    
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("GEMINI_API_KEY environment variable not found. Testing fallback regex methods only.")
    
    processor = NLPOrderProcessor(gemini_api_key=gemini_api_key)
    
    test_queries = [
        "Need 50 kg tomatoes and 30 bottles milk urgently by tomorrow",
        "Looking for organic vegetables, about 20 boxes, budget $500, deliver by Friday",
        "We need bulk dairy products - 100 units cheese, 200 bottles milk, no rush"
    ]
    
    for query in test_queries:
        result = processor.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Structured: {result}")

if __name__ == "__main__":
    
    test_nlp_processing()
    
   