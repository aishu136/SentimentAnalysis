import re
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm

class PositiveParaphraser:
    def __init__(self, model_name="t5-small", num_epochs=3, learning_rate=5e-5):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    @staticmethod
    def clean_text(text):
        # Remove HTML tags, special characters, and extra whitespaces
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        return ' '.join(text.split())

    def preprocess_positive_data(self):
        imdb = load_dataset("imdb")
        positive_reviews = [review['text'] for review, label in zip(imdb['train'], imdb['train']['label']) if label == 1]
        cleaned_positive_data = [self.clean_text(text) for text in positive_reviews]
        return cleaned_positive_data



    def fine_tune_on_positive_data(self):
        positive_data = self.preprocess_positive_data()
        tokenized_positive_data = self.tokenizer(positive_data, return_tensors='pt', padding=True, truncation=True)
        labels = self.tokenizer(positive_data, return_tensors='pt', padding=True, truncation=True)['input_ids']

        train_dataset = TensorDataset(tokenized_positive_data['input_ids'], tokenized_positive_data['attention_mask'], labels)
        train_data = DataLoader(train_dataset, batch_size=4, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.model.train()
            for batch in tqdm(train_data, desc=f"Epoch {epoch + 1}"):
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        # Print average loss after each epoch
            avg_loss = loss.item() / len(train_data)
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")


    def paraphrase_positive(self, text):
        cleaned_text = self.clean_text(text)
        input_ids = self.tokenizer.encode(cleaned_text, return_tensors="pt")
        paraphrased_ids = self.model.generate(input_ids, max_length=1000, num_beams=5, temperature=0.8,top_k=50)
        paraphrased_text = self.tokenizer.decode(paraphrased_ids[0], skip_special_tokens=True,do_sample=True)
        return paraphrased_text

    def save_model(self, model_path="positive_paraphraser"):
        # Save the model to the specified directory
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def load_model(self, model_path="positive_paraphraser"):
        # Load the model from the specified directory
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)


def main():
    paraphraser = PositiveParaphraser()

    # Uncomment the following line if you want to fine-tune the model on positive data
    paraphraser.fine_tune_on_positive_data()

    # Save the model after fine-tuning
    paraphraser.save_model()

    # Load the model
    paraphraser.load_model()

    # Example usage
    input_text = "They were great friends, and their friendship blossomed over time."
    paraphrased_text = paraphraser.paraphrase_positive(input_text)
    print("Original:", input_text)
    print("Paraphrased:", paraphrased_text)


if __name__ == "__main__":
    main()
