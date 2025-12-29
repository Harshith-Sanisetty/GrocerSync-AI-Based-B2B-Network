from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import schemas
from ..database import get_db
from ..ml.forecast import predict_shortage

router = APIRouter(
    prefix="/ml",
    tags=["ml"],
)

@router.get("/predict/{product_id}", response_model=schemas.PredictionResponse)
def get_prediction(product_id: int, db: Session = Depends(get_db)):
    result = predict_shortage(db, product_id)
    if not result:
        raise HTTPException(status_code=404, detail="Product not found or prediction failed")
    return result
