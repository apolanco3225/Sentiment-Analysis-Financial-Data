import gradio as gr
from model_config import SentimentAnalyzer
import torch

class SentimentAnalysisApp:
    def __init__(self, model_path="./models"):
        self.analyzer = SentimentAnalyzer()
        self.model, self.tokenizer = self.analyzer.load_saved_model(model_path)
        self.model.eval()

    def predict(self, text):
        """Make prediction for a given text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            
        sentiment = self.analyzer.id2label_dict[predicted_class]
        confidence = predictions[0][predicted_class].item()
        
        return {
            "sentiment": sentiment,
            "confidence": f"{confidence:.2%}"
        }

def create_app():
    app = SentimentAnalysisApp()
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=app.predict,
        inputs=gr.Textbox(
            lines=3,
            placeholder="Enter financial text here...",
            label="Financial Text"
        ),
        outputs=gr.JSON(label="Sentiment Analysis Results"),
        title="Financial Text Sentiment Analysis",
        description="""
        This app analyzes the sentiment of financial text using a fine-tuned DistilBERT model.
        Enter any financial text to get the sentiment (positive, neutral, or negative) and confidence score.
        """,
        examples=[
            ["The company reported strong quarterly earnings, exceeding market expectations."],
            ["The stock price remained unchanged during the trading session."],
            ["The company announced significant layoffs due to declining revenue."]
        ]
    )
    
    return interface

if __name__ == "__main__":
    interface = create_app()
    interface.launch() 