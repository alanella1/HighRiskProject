import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split


class PostDatasetForFineTuning(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        selftext = row['selftext']
        label = 1.0 - row['last_post']

        # Tokenize
        tokens = self.tokenizer(
            selftext,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Return input IDs, attention mask, and label
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'label': torch.tensor(int(label), dtype=torch.long)
        }


class WeightedSequenceClassifier(nn.Module):
    def __init__(self, ):
        super(WeightedSequenceClassifier, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/MiniLM-L12-H384-uncased",
            num_labels=1
        )
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((5.6397 * .85), dtype=torch.float32))

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        logits = outputs.logits

        if labels is not None:
            labels = labels.float()
            loss = self.loss_fn(logits.squeeze(), labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}


def post_prediction_training(dataset, model, batch_size=32, lr=0.001, verbose=False):
    # Split the dataset
    train_indices, test_indices = train_test_split(
        list(range(len(dataset))),
        test_size=0.1,
        random_state=42
    )
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # Training loop
    for epoch in range(1):
        model.train()
        running_training_loss = 0.0
        n = 0
        progress_bar = tqdm(train_loader, desc="Training..", unit="batch")
        for batch in progress_bar:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['label'].cuda()

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']

            loss.backward()
            optimizer.step()

            running_training_loss += loss.item()
            n += 1
            progress_bar.set_postfix(loss=(running_training_loss / n))

        if verbose:
            print(f"Epoch {epoch + 1}, Loss: {running_training_loss / n}")
    # TRAINING LOOP OVER

    # Evaluation
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    n = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['label'].cuda()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            test_loss += outputs['loss'].item()
            n += 1

            probs = torch.sigmoid(outputs['logits'])
            predictions = (probs >= 0.5).long()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Store all concerned stats
    avg_test_loss = test_loss / n
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(
        f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
        f"F1: {f1:.4f}")


model_name = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = WeightedSequenceClassifier()
df = pd.read_csv('depression.csv')
dataset = PostDatasetForFineTuning(df, tokenizer)
post_prediction_training(dataset, model, batch_size=64, lr=1e-5, verbose=True)
