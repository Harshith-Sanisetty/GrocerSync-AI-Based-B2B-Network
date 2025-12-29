from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas, database
from ..database import get_db

router = APIRouter(
    prefix="/inventory",
    tags=["inventory"],
)

@router.get("/", response_model=List[schemas.Product])
def get_inventory(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    products = db.query(models.Product).offset(skip).limit(limit).all()
    return products

@router.post("/", response_model=schemas.Product)
def create_product(product: schemas.ProductCreate, store_id: int, db: Session = Depends(get_db)):
    db_product = models.Product(**product.dict(), store_id=store_id)
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product

@router.put("/{product_id}", response_model=schemas.Product)
def update_product(product_id: int, product: schemas.ProductCreate, db: Session = Depends(get_db)):
    db_product = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    for key, value in product.dict().items():
        setattr(db_product, key, value)
    
    db.commit()
    db.refresh(db_product)
    return db_product

@router.delete("/{product_id}")
def delete_product(product_id: int, db: Session = Depends(get_db)):
    db_product = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    db.delete(db_product)
    db.commit()
    return {"ok": True}

@router.post("/update/{product_id}", response_model=schemas.InventoryLog)
def update_stock(product_id: int, update: schemas.StockUpdateRequest, db: Session = Depends(get_db)):
    product = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    # Update current stock
    new_stock = max(0.0, (product.current_stock or 0.0) - (update.sold_qty or 0.0) + (update.restocked_qty or 0.0))
    product.current_stock = new_stock
    # Create log
    log = models.InventoryLog(
        product_id=product.id,
        sold_qty=update.sold_qty or 0.0,
        restocked_qty=update.restocked_qty or 0.0,
        remaining_stock=new_stock,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log

@router.get("/logs/{product_id}", response_model=List[schemas.InventoryLog])
def get_logs(product_id: int, limit: int = 30, db: Session = Depends(get_db)):
    product = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    logs = (
        db.query(models.InventoryLog)
        .filter(models.InventoryLog.product_id == product_id)
        .order_by(models.InventoryLog.date.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(logs))
