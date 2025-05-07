# Financial Text Sentiment Analysis

This project implements a sentiment analysis model for financial text using a fine-tuned DistilBERT model with LoRA (Low-Rank Adaptation) for efficient fine-tuning. The model is trained on the Auditor Sentiment dataset and can classify financial text into three categories: positive, neutral, or negative.

## Project Structure

```
.
├── notebooks/
│   └── baseline.ipynb          # Original notebook with model development
├── src/
│   ├── model_config.py         # Model configuration and training logic
│   └── app.py                  # Gradio web interface for model deployment
├── models/                     # Directory for saved models (created during training)
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Features

- Fine-tuned DistilBERT model for financial text sentiment analysis
- Parameter-Efficient Fine-Tuning using LoRA
- Web interface using Gradio for easy interaction
- Support for three sentiment classes: positive, neutral, and negative
- Confidence scores for predictions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Sentiment-Analysis-Financial-Data.git
cd Sentiment-Analysis-Financial-Data
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model:

```bash
python src/model_config.py
```

This will:
1. Load the Auditor Sentiment dataset
2. Initialize the DistilBERT model
3. Configure and apply LoRA
4. Train the model
5. Save the trained model to the `models/` directory

### Running the Web Interface

To start the Gradio web interface:

```bash
python src/app.py
```

This will launch a web interface where you can:
- Enter financial text for sentiment analysis
- Get predictions with confidence scores
- Try example inputs

## Model Details

- Base Model: DistilBERT (uncased)
- Fine-tuning Method: LoRA (Low-Rank Adaptation)
- Number of Classes: 3 (positive, neutral, negative)
- Training Dataset: Auditor Sentiment Dataset
- Evaluation Metric: Accuracy

## Performance

The model achieves approximately 85% accuracy on the test set, demonstrating strong performance in classifying financial text sentiment.

## License

This project is licensed under the MIT License - see the LICENSE file for details.