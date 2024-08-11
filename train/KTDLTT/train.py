import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# Load data
data =  pd.read_excel('news_data1.xlsx')

# Map labels to 0 and 1
# data['labels'] = data['labels'].map({'unreliable': 0, 'reliable': 1})

# Initialize RoBERTa tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

# Custom dataset class
class NewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data.dropna(subset=['labels'])  # Drop rows with missing labels
        if len(self.data) == 0:
            raise ValueError("No valid samples found in the dataset.")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title = str(self.data['title'].iloc[idx])  # Convert to string
        content = str(self.data['content'].iloc[idx])  # Convert to string
        label = int(self.data['labels'].iloc[idx])  # Convert to integer

        input_text = title + " " + content

        encoding = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }



# Define max sequence length
max_length = 512

# Create dataset and dataloader
dataset = NewsDataset(data, tokenizer, max_length)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Fine-tune RoBERTa for sequence classification
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=2)

# Define optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Define training parameters
epochs = 3

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        print(batch, 'batch')
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
    
    avg_loss = running_loss / len(train_loader)
    print(f'Training Loss: {avg_loss:.4f}')
    # Adjust learning rate
    scheduler.step()

# Evaluate the model
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        

model.save_pretrained('saved_model3')
# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Unreliable", "Reliable"]))

# Print metrics
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"Micro F1 Score: {micro_f1:.4f}")
print(f"Weighted F1 Score: {weighted_f1:.4f}")


