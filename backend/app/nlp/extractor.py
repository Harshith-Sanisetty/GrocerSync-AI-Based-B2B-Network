import re
from typing import List, Dict, Any

def extract_order_from_text(text: str) -> Dict[str, Any]:
    """
    Extracts order items from natural language text.
    Example: "I need 5 kg rice & 2 oil cans tomorrow morning."
    Output: {
       "items": [{"product":"Rice","qty":5,"unit":"kg"}, ...],
       "delivery": "tomorrow morning"
    }
    """
    text = text.lower()
    items = []
    
    # Pattern: number space? unit space? product
    # Units: kg, g, l, litre, liter, packet, pkt, box, can, piece, pc
    # Regex: (\d+)\s*(kg|g|l|litre|liter|packet|pkt|box|can|piece|pc|cans|packets|boxes)\s+([a-z]+)
    
    # We also need to handle "5 rice" (implied unit or count)
    
    # Strategy: Split by '&', 'and', ',' to get chunks
    chunks = re.split(r'[,&]|\band\b', text)
    
    delivery_keywords = ["tomorrow", "today", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "morning", "evening", "afternoon"]
    delivery_info = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk: continue
        
        # Check for delivery keywords
        is_delivery = False
        for kw in delivery_keywords:
            if kw in chunk:
                delivery_info.append(chunk)
                is_delivery = True
                break
        if is_delivery:
            continue

        # Try to parse item
        # Look for number
        match = re.search(r'(\d+)\s*([a-zA-Z]+)?\s*(.*)', chunk)
        if match:
            qty = float(match.group(1))
            unit_or_product_part1 = match.group(2)
            product_part2 = match.group(3)
            
            unit = "units"
            product = ""
            
            known_units = ["kg", "kgs", "g", "gm", "l", "ltr", "can", "cans", "pkt", "packet", "packets", "box", "boxes", "pc", "pcs"]
            
            if unit_or_product_part1 and unit_or_product_part1 in known_units:
                unit = unit_or_product_part1
                product = product_part2
            elif unit_or_product_part1:
                # unit might be part of product if not a known unit
                # e.g. "5 apples" -> qty=5, unit_part=apples, prod_part=""
                # But regex above captures (\d+) (word)? (rest)
                if not product_part2:
                    product = unit_or_product_part1
                else:
                    product = f"{unit_or_product_part1} {product_part2}"
            
            if product:
                items.append({
                    "product": product.strip().title(),
                    "qty": qty,
                    "unit": unit
                })

    return {
        "items": items,
        "delivery": " ".join(delivery_info) if delivery_info else "As soon as possible"
    }
