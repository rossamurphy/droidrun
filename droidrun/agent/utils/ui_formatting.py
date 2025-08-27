"""
Shared UI formatting utilities for consistent UI element representation.

This module provides unified UI formatting functions used by both:
1. Inference-time context processing (chat_utils)
2. Training data collection (conversation collection)

This ensures identical formatting between training and inference.
"""

def format_ui_elements_as_text(elements) -> str:
    """Convert UI elements to clean text representation with essential info preserved.
    
    This function handles both:
    - Live UI state from Android uiautomator (inference time)  
    - Saved UI elements from _elements.json files (training data collection)
    
    Args:
        elements: UI elements data (list, dict, or JSON string)
        
    Returns:
        str: Formatted UI elements text in the format:
        UI Elements:
          [0] "Text" id:resource_id (clickable) at bounds
          [1] [ClassName] class:ClassName at bounds
    """
    if not elements:
        return "No UI elements detected"

    # Handle both list and dict formats
    if isinstance(elements, dict):
        elements = elements.get('elements', [elements])
    elif isinstance(elements, str):
        try:
            import json
            parsed = json.loads(elements)
            if isinstance(parsed, dict):
                elements = parsed.get('elements', [parsed])
            else:
                elements = parsed
        except (json.JSONDecodeError, TypeError):
            return "UI Elements:\n  (Unable to parse UI state)"

    if not isinstance(elements, list):
        elements = [elements]

    ui_text = "UI Elements:\n"
    for elem in elements:
        if not isinstance(elem, dict):
            continue
            
        idx = elem.get("index", "?")
        text = elem.get("text", "")
        desc = elem.get("content-desc", "")
        
        # Get both class and className fields (different data sources use different names)
        cls = elem.get("class", elem.get("className", ""))
        cls_short = cls.split(".")[-1] if cls else ""
        
        # Handle both resource-id and resourceId field names
        resource_id = elem.get("resource-id", elem.get("resourceId", ""))
        bounds = elem.get("bounds", "")
        clickable = elem.get("clickable", False)

        # Build comprehensive label with essential identifying information
        label_parts = []
        if text:
            label_parts.append(f'"{text}"')
        elif desc:
            label_parts.append(f'"{desc}"')
        else:
            label_parts.append(f"[{cls_short}]" if cls_short else "[Element]")
            
        # Add resource ID for better identification (crucial for buttons)
        if resource_id:
            label_parts.append(f"id:{resource_id}")
            
        # Add class info for context
        if cls_short and not resource_id:  # Only show class if no resource_id
            label_parts.append(f"class:{cls_short}")

        label = " ".join(label_parts)
        
        if clickable:
            ui_text += f"  [{idx}] {label} (clickable) at {bounds}\n"
        else:
            ui_text += f"  [{idx}] {label} at {bounds}\n"

    return ui_text