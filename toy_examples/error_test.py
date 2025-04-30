from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer_error = AutoTokenizer.from_pretrained("byviz/bylastic_classification_logs")
model_error = AutoModelForSequenceClassification.from_pretrained("byviz/bylastic_classification_logs")


text = "[ERROR] Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key"
inputs = tokenizer_error(text, return_tensors="pt")

with torch.no_grad():
    outputs = model_error(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1)
    confidence = probabilities[0][prediction.item()].item()
    
print(f"Prediction: {model_error.config.id2label[prediction.item()]} (Class {prediction.item()}), Confidence: {confidence:.4f}")
