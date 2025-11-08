import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np



class NumpyDataset(Dataset):
    def __init__(self, images, labels, processor):
        self.images = images
        self.labels = labels
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transform(img)
        lbl = torch.tensor(self.labels[idx], dtype=torch.float32)  # one-hot vector
        return img, lbl

class ViTTrainer:
    def __init__(self, model_name="google/vit-base-patch16-224", num_classes=10, lr=1e-4, batch_size=16, epochs=15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        # Processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)

        # Freeze ViT base layers
        for param in self.model.vit.parameters():
            param.requires_grad = False

        # Replace classifier
        self.model.classifier = nn.Linear(self.model.config.hidden_size, num_classes).to(self.device)

        # BCEWithLogitsLoss for one-hot labels
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    # Prepare train/validation loaders
    def prepare_data(self, data, labels, split_ratio=0.8):
        dataset = NumpyDataset(data, labels, self.processor)
        train_size = int(split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        print(f"Data prepared — Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Training loop
    def train(self):
        print("Starting training...")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)  # shape: (batch_size, num_classes)
                
                self.optimizer.zero_grad()
                outputs = self.model(images).logits
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}] — Loss: {avg_loss:.4f}")
        print("Training completed!")

    # Evaluation loop
    def evaluate(self):
        print("Evaluating model...")
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images).logits
                probs = torch.sigmoid(outputs)
                predicted = torch.argmax(probs, dim=1)
                true_class = torch.argmax(labels, dim=1)
                total += labels.size(0)
                correct += (predicted == true_class).sum().item()
        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")
        return acc

    # Predict probabilities and top class for a single image
    def predict(self, image):
        self.model.eval()
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])
        img_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor).logits
            probs = torch.sigmoid(outputs)
            top_prob, top_class = torch.max(probs, dim=1)
        
        return probs.cpu().numpy().flatten(), int(top_class), float(top_prob)

    # Save model
    def save(self, path="vit_model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    # Load model
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

