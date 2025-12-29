from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas
from ..database import get_db
from ..nlp.extractor import extract_order_from_text

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

@router.post("/parse", response_model=schemas.NLPResponse)
def parse_chat_message(request: schemas.NLPRequest):
    result = extract_order_from_text(request.text)
    return result

@router.post("/message", response_model=schemas.Message)
def send_message(msg: schemas.MessageCreate, db: Session = Depends(get_db)):
    db_msg = models.Message(**msg.dict())
    db.add(db_msg)
    db.commit()
    db.refresh(db_msg)
    return db_msg

@router.get("/history/{order_id}", response_model=List[schemas.Message])
def get_chat_history(order_id: int, db: Session = Depends(get_db)):
    return db.query(models.Message).filter(models.Message.order_id == order_id).order_by(models.Message.timestamp).all()
