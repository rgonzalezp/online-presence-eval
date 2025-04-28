# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Minos-v1")
model = AutoModelForSequenceClassification.from_pretrained(
    "NousResearch/Minos-v1",
    num_labels=2,
    id2label={0: "Non-refusal", 1: "Refusal"},  # Explicitly set label names
    label2id={"Non-refusal": 0, "Refusal": 1}
)

# Format input
# text = "<|user|>\n# Mode: no-search Use only your internal knowledge. \# Tools: Do not use any tools to display information Tell me everything you know about user experience researcher Ricardo Penuela.\n<|assistant|>\nBased on my internal knowledge up to June 2024, there is no widely recognized or notable user experience researcher named Ricardo Penuela in the mainstream UX research community, academic publications, or major industry events. It is possible that Ricardo Penuela may be an emerging professional, a local practitioner, or someone whose work has not yet gained broad visibility in global UX circles. If you have more specific context—such as a company, project, or publication associated with Ricardo Penuela—please provide it, and I can try to offer more targeted information or context based on what is generally known about UX research roles and contributions. Otherwise, there is no significant public information about a UX researcher by that name in the major sources available up to my knowledge cutoff."

text = "<|user|>\n# Mode: web-search You may search the web before answering. # Tools: You may use additional tools to support your search. Tell me everything you know about accessibility researcher Ricardo Enrique Gonzalez Penuela.\n<|assistant|>\nRicardo Enrique Gonzalez Penuela is an accessibility and Human-Computer Interaction (HCI) researcher. He is currently a PhD Candidate in the Information Science Department at Cornell University, specifically at Cornell Tech in New York City. His research focuses on designing intelligent interactive systems with the goal of empowering people with disabilities. A significant portion of his work involves understanding how people with visual impairments, including those who are blind and have low vision, interact with and utilize AI services and other technologies. Some key areas of his research include: *   Understanding how people with disabilities use AI services to inform improvements for empowering them. *   Creating equitable and accessible Extended Reality (XR) experiences for people with visual impairments. *   Improving the accessibility of 3D virtual environments for blind and low vision users. *   Investigating the use and challenges of AI-powered scene description applications for blind and low vision individuals."
inputs = tokenizer(text, return_tensors="pt")

# Get prediction
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1)
    confidence = probabilities[0][prediction.item()].item()
    
print(f"Prediction: {model.config.id2label[prediction.item()]} (Class {prediction.item()}), Confidence: {confidence:.4f}")