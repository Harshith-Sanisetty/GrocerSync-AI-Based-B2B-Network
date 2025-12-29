from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine, Base, SessionLocal
from . import models
from .routers import inventory, suppliers, orders, chat, ml

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="GrocerSync API", version="1.0.0")

# Seed minimal data for demo
def seed_data():
    db = SessionLocal()
    try:
        # Seed a store
        if db.query(models.Store).count() == 0:
            store = models.Store(name="Demo Store", owner_email="owner@example.com", firebase_uid="store_demo", address="Main Street")
            db.add(store)
            db.commit()
        # Seed suppliers
        if db.query(models.Supplier).count() == 0:
            s1 = models.Supplier(business_name="Wholesale Mart", contact="+1 234 567 890", address="123 Market St", rating=4.8, email="wholesale@example.com")
            s2 = models.Supplier(business_name="Fresh Farms", contact="+1 987 654 321", address="45 Green Rd", rating=4.5, email="fresh@example.com")
            db.add_all([s1, s2])
            db.commit()
        # Seed products
        store = db.query(models.Store).first()
        if store and db.query(models.Product).count() == 0:
            p1 = models.Product(store_id=store.id, name="Basmati Rice", category="Grains", unit="kg", current_stock=45, min_threshold=50)
            p2 = models.Product(store_id=store.id, name="Sunflower Oil", category="Oil", unit="L", current_stock=12, min_threshold=20)
            db.add_all([p1, p2])
            db.commit()
    finally:
        db.close()

seed_data()

# CORS
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inventory.router)
app.include_router(suppliers.router)
app.include_router(orders.router)
app.include_router(chat.router)
app.include_router(ml.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to GrocerSync API"}
