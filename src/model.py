import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


class TransformerModel:
    """
    DistilBERT-based sentiment classifier.

    Model:
        distilbert-base-uncased-finetuned-sst-2-english

    The model is fine-tuned on SST-2 for binary sentiment classification.
    """

    def __init__(self):
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            output_attentions=True
        )

        self.model.eval()

        self.id2label = self.model.config.id2label

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        predicted_id = int(
            torch.argmax(probabilities).item()
        )

        confidence = float(
            probabilities[predicted_id].item() * 100
        )

        sentiment = self.id2label[predicted_id]

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "inputs": inputs,
            "outputs": outputs
        }

    def get_tokens(self, inputs):
        input_ids = inputs["input_ids"][0]

        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids
        )

        return tokens
