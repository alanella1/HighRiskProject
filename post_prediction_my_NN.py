from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset


class PostDataset(Dataset):
    def __init__(self, dataset_path):
        # Load the dataset
        self.dataset = torch.load(dataset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve a single sample
        sample = self.dataset[idx]
        selftext = sample['selftext']
        label = 1.0 - sample['label']
        return selftext, label


class LinearCell(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearCell, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PostClassifier(nn.Module):
    def __init__(self, input_size):
        super(PostClassifier, self).__init__()
        self.drop_rate = 0.2
        self.fc1 = LinearCell(input_size, 64)
        self.fc2 = LinearCell(64, 32)
        self.fc3 = LinearCell(32, 16)
        self.final = nn.Linear(16, 1)
        self.drop_1 = nn.Dropout(self.drop_rate)
        self.drop_2 = nn.Dropout(self.drop_rate)
        self.drop_3 = nn.Dropout(self.drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop_1(x)
        x = self.fc2(x)
        x = self.drop_2(x)
        x = self.fc3(x)
        x = self.drop_3(x)
        return self.final(x)


def k_fold_cross_validation(dataset, k=10, batch_size=32, lr=0.001, verbose=False):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    losses = []
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for fold, (train_indices, test_indices) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{k}")

        # Split the dataset
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # Initialize the model, loss function, and optimizer
        model = PostClassifier(384).cuda()

        pos_weight = torch.tensor(5.6397, dtype=torch.float32).cuda()  # Calculated earlier
        alpha = 0.5  # tuneable parameter
        pos_weight = pos_weight * alpha
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        # Training loop
        for epoch in range(1000):
            model.train()
            running_training_loss = 0.0
            n = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                loss.backward()
                optimizer.step()

                running_training_loss += loss.item()
                n += 1

            if verbose:
                print(f"Epoch {epoch + 1}, Loss: {running_training_loss / n}")

            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.view(-1, 1))
                    test_loss += loss.item()

            avg_test_loss = test_loss / len(test_loader)
            scheduler.step(avg_test_loss)
            if scheduler.get_last_lr()[0] < lr * 1e-3:  # If dropped three times, done training
                break
        # TRAINING LOOP OVER

        # Evaluation
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                test_loss += loss.item()

                outputs = torch.sigmoid(outputs)
                predictions = (outputs >= 0.5).float()  # Threshold at 0.5 for binary classification
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Store all the concerned stats
        avg_test_loss = test_loss / len(test_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        losses.append(avg_test_loss)
        metrics["accuracy"].append(accuracy)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)

        print(
            f"Fold {fold + 1} Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
            f"F1: {f1:.4f}")

    # Return the average loss across all folds
    mean_loss = np.mean(losses)
    mean_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
    print(f"Mean Test Loss (1 fold): {mean_loss:.4f}")
    print(
        f"Mean Test Metrics (1 fold): Accuracy: {mean_metrics['accuracy']:.4f}, Precision: {mean_metrics['precision']:.4f}, Recall: "
        f"{mean_metrics['recall']:.4f}, F1: {mean_metrics['f1']:.4f}")
    return mean_loss, mean_metrics


dataset_path = 'standardized_encodings_dataset.pt'
post_dataset = PostDataset(dataset_path)
k_fold_cross_validation(post_dataset, k=10, batch_size=128, lr=1e-2, verbose=True)
