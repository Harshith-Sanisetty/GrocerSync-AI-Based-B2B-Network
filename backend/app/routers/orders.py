from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas
from ..database import get_db

router = APIRouter(
    prefix="/orders",
    tags=["orders"],
)

@router.post("/", response_model=schemas.Order)
def create_order(order: schemas.OrderCreate, store_id: int, db: Session = Depends(get_db)):
    # Calculate amount (mock logic, ideally fetch prices from supplier_products)
    amount = sum(item.qty * (item.price or 0) for item in order.items)
    
    db_order = models.Order(
        store_id=store_id,
        supplier_id=order.supplier_id,
        items=[item.dict() for item in order.items],
        amount=amount,
        delivery_date=order.delivery_date
    )
    db.add(db_order)
    db.commit()
    db.refresh(db_order)
    return db_order

@router.get("/", response_model=List[schemas.Order])
def get_orders(store_id: int, db: Session = Depends(get_db)):
    return db.query(models.Order).filter(models.Order.store_id == store_id).all()
