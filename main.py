# Tested with Python Version 3.8 on 10/15/2023

# Install PyTorch with CUDA support (assuming CUDA 10.2, but adjust as per your CUDA version)
#  conda install pytorch
#  torchvision torchaudio cudatoolkit=10.2 -c pytorch

# Install transformers and datasets libraries using pip within the conda environment
# pip install transformers datasets

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch

# Load data from Hugging Face datasets
wikitext_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
texts = wikitext_dataset["train"]["text"]

# Define dataset
class FineTuningDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.tokenizer(self.texts[index], return_tensors="pt", truncation=True, max_length=self.max_length)["input_ids"].squeeze()

# Load pretrained model and tokenizer
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Create data loader
dataset = FineTuningDataset(tokenizer, texts)
dataloader = DataLoader(dataset, shuffle=True, batch_size=8)

# Set device, optimizer, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader))

# Fine-tuning loop
NUM_EPOCHS = 3
for epoch in range(NUM_EPOCHS):
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = batch.to(device)
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_minstral")
