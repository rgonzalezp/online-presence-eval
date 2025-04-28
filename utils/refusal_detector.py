# refusal_detector.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load once on import
_TOKENIZER = AutoTokenizer.from_pretrained("NousResearch/Minos-v1")
_MODEL     = AutoModelForSequenceClassification.from_pretrained(
    "NousResearch/Minos-v1",
    num_labels=2,
    id2label={0: "Non-refusal", 1: "Refusal"},
    label2id={"Non-refusal": 0, "Refusal": 1}
)
_MODEL.eval()

def detect_refusal(user_query: str, text_response: str) -> tuple[str, float]:
    """
    Returns:
      (label, confidence)
    Internally prepends the special markers before classification.
    """
    # build the combined string
    dialogue = (
        f"<|user|>{user_query} "
        f"<|assistant|>{text_response}"
    )

    inputs = _TOKENIZER(dialogue, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs    = _MODEL(**inputs)
        probs      = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx   = int(torch.argmax(probs, dim=-1))
        label      = _MODEL.config.id2label[pred_idx]
        confidence = float(probs[0][pred_idx])
    return label, confidence
