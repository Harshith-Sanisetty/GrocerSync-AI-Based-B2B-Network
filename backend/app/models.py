from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class Store(Base):
    __tablename__ = "stores"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    owner_email = Column(String, unique=True, index=True)
    firebase_uid = Column(String, unique=True, index=True) # Link to auth
    address = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    products = relationship("Product", back_populates="store")
    orders = relationship("Order", back_populates="store")

class Supplier(Base):
    __tablename__ = "suppliers"
    id = Column(Integer, primary_key=True, index=True)
    business_name = Column(String, index=True)
    contact = Column(String)
    address = Column(String)
    rating = Column(Float, default=0.0)
    firebase_uid = Column(String, unique=True, index=True) # Link to auth if supplier logs in
    email = Column(String, unique=True)

    catalog = relationship("SupplierProduct", back_populates="supplier")
    orders = relationship("Order", back_populates="supplier")

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.id"))
    name = Column(String, index=True)
    category = Column(String, index=True)
    unit = Column(String)
    current_stock = Column(Float, default=0.0)
    min_threshold = Column(Float, default=10.0)
    expiry_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    store = relationship("Store", back_populates="products")
    logs = relationship("InventoryLog", back_populates="product")

class InventoryLog(Base):
    __tablename__ = "inventory_logs"
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    date = Column(DateTime(timezone=True), server_default=func.now())
    sold_qty = Column(Float, default=0.0)
    restocked_qty = Column(Float, default=0.0)
    remaining_stock = Column(Float)

    product = relationship("Product", back_populates="logs")

class SupplierProduct(Base):
    __tablename__ = "supplier_products"
    id = Column(Integer, primary_key=True, index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"))
    product_name = Column(String, index=True)
    price = Column(Float)
    min_order_qty = Column(Float, default=1.0)

    supplier = relationship("Supplier", back_populates="catalog")

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.id"))
    supplier_id = Column(Integer, ForeignKey("suppliers.id"))
    status = Column(String, default="Requested") # Requested, Negotiation, Agreed, Confirmed, Dispatched, Delivered
    items = Column(JSON) # List of {product, qty, unit, price}
    amount = Column(Float, default=0.0)
    delivery_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    store = relationship("Store", back_populates="orders")
    supplier = relationship("Supplier", back_populates="orders")
    messages = relationship("Message", back_populates="order")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    sender_id = Column(String) # Firebase UID or 'system'
    message = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    order = relationship("Order", back_populates="messages")
