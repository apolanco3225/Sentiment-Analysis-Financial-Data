import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label_dict = {
            2: 'positive',
            1: 'neutral',
            0: 'negative'
        }
        self.label2id_dict = {
            'positive': 2,
            'neutral': 1,
            'negative': 0
        }
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.id2label_dict,
            label2id=self.label2id_dict
        )
        
        # Freeze base model parameters
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def load_data(self, test_size=0.2, seed=23):
        """Load and preprocess the dataset"""
        dataset = load_dataset(
            "FinanceInc/auditor_sentiment",
            split="train"
        ).train_test_split(
            test_size=test_size,
            shuffle=True,
            seed=seed
        )
        
        # Tokenize datasets
        tokenized_dataset = {}
        for split in ["train", "test"]:
            tokenized_dataset[split] = dataset[split].map(
                lambda x: self.tokenizer(x["sentence"], truncation=True),
                batched=True
            )
        
        return tokenized_dataset

    def setup_lora(self, r=16, lora_alpha=32, lora_dropout=0.05):
        """Configure and setup LoRA"""
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_lin", "k_lin", "v_lin"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, config)
        return self.model

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": (predictions == labels).mean()}

    def train(self, tokenized_dataset, output_dir="./models", num_epochs=20):
        """Train the model"""
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=output_dir,
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                num_train_epochs=num_epochs,
                weight_decay=0.01,
                load_best_model_at_end=True,
            ),
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        return trainer

    def save_model(self, output_dir="./models"):
        """Save the model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_saved_model(self, model_path):
        """Load a saved model"""
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        return self.model, self.tokenizer

if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    tokenized_dataset = analyzer.load_data()
    analyzer.setup_lora()
    trainer = analyzer.train(tokenized_dataset)
    analyzer.save_model() 