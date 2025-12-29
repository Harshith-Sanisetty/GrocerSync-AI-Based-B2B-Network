from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime

# --- Product Schemas ---
class ProductBase(BaseModel):
    name: str
    category: str
    unit: str
    current_stock: float
    min_threshold: float
    expiry_date: Optional[datetime] = None

class ProductCreate(ProductBase):
    pass

class Product(ProductBase):
    id: int
    store_id: int
    created_at: datetime

    class Config:
        orm_mode = True

# --- Supplier Schemas ---
class SupplierBase(BaseModel):
    business_name: str
    contact: str
    address: str
    email: str

class SupplierCreate(SupplierBase):
    pass

class Supplier(SupplierBase):
    id: int
    rating: float
    firebase_uid: Optional[str] = None

    class Config:
        orm_mode = True

# --- Order Schemas ---
class OrderItem(BaseModel):
    product: str
    qty: float
    unit: str
    price: Optional[float] = 0.0

class OrderBase(BaseModel):
    supplier_id: int
    items: List[OrderItem]
    delivery_date: Optional[datetime] = None

class OrderCreate(OrderBase):
    pass

class Order(OrderBase):
    id: int
    store_id: int
    status: str
    amount: float
    created_at: datetime

    class Config:
        orm_mode = True

# --- Chat Schemas ---
class MessageBase(BaseModel):
    message: str

class MessageCreate(MessageBase):
    order_id: int
    sender_id: str

class Message(MessageBase):
    id: int
    order_id: int
    sender_id: str
    timestamp: datetime

    class Config:
        orm_mode = True

# --- NLP Schemas ---
class NLPRequest(BaseModel):
    text: str

class NLPResponse(BaseModel):
    items: List[OrderItem]
    delivery: Optional[str] = None

# --- ML Schemas ---
class PredictionRequest(BaseModel):
    product_id: int

class PredictionResponse(BaseModel):
    product_id: int
    predicted_daily_usage: float
    run_out_in_days: float
    recommended_restock: float
    confidence_score: float

# --- Inventory Log Schemas ---
class InventoryLogBase(BaseModel):
    product_id: int
    sold_qty: float = 0.0
    restocked_qty: float = 0.0
    remaining_stock: float

class InventoryLogCreate(InventoryLogBase):
    pass

class InventoryLog(InventoryLogBase):
    id: int
    date: datetime

    class Config:
        orm_mode = True

class StockUpdateRequest(BaseModel):
    sold_qty: float = 0.0
    restocked_qty: float = 0.0
