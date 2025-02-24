from tqdm import tqdm
import numpy as np
from collections import defaultdict

import torch
from torch import nn

class NearestClassCenterClassifier(nn.Module):
    def __init__(self, model, device):
        super(NearestClassCenterClassifier, self).__init__()
        self.model = model.to(device)
        self.device = device

        self.class_centers = {}

    def forward(self, dataloader):
        self.fit(dataloader)
        accuracy, error_rate = self.evaluate(dataloader)
        print(f"Accuracy: {accuracy:.2f}, Error Rate: {error_rate:.2f}")
        return accuracy, error_rate

    @torch.no_grad()
    def fit(self, dataloader):
        class_features = defaultdict(list)
        self.model.eval()

        for images, labels in tqdm(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            features = self.model(images)
            for feature, label in zip(features, labels):
                class_features[label.item()].append(feature.cpu().numpy())

        self.class_centers = {cls: np.mean(features, axis=0) for cls, features in class_features.items()}

    @torch.no_grad()
    def predict_batch(self, images):
        self.model.eval()
        batch_predictions = []

        features = self.model(images)
        for feature in features:
            distances = {cls: np.linalg.norm(feature.cpu().numpy() - center)
                         for cls, center in self.class_centers.items()}
            predicted_class = min(distances, key=distances.get)
            batch_predictions.append(predicted_class)
        return torch.tensor(batch_predictions)

    @torch.no_grad()
    def evaluate(self, dataloader):
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            predictions = self.predict_batch(images)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predictions.cpu() == labels.cpu()).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        error_rate = 1 - accuracy
        return accuracy, error_rate
