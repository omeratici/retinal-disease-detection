# Gerekli kütüphaneler
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
# ---------------------------------
# Kaynak kullanımı ölçümü için
# ---------------------------------
import psutil
import subprocess
import time
def log_resource_usage():
    """
    CPU, RAM ve GPU bellek kullanımını raporlar.
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0
    print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")
    print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} MB, "
          f"GPU Memory Reserved: {gpu_memory_reserved:.2f} MB")

def get_gpu_power_usage():
    """
    NVIDIA GPU gücünü (Watt) ölçer. nvidia-smi çıktısını parse eder.
    """
    if torch.cuda.is_available():
        try:
            result = subprocess.check_output(["nvidia-smi",
                                              "--query-gpu=power.draw",
                                              "--format=csv,nounits,noheader"])
            power_usage = float(result.decode().strip())
            print(f"GPU Power Usage: {power_usage:.2f} W")
        except Exception as e:
            print("GPU power usage ölçümünde hata oluştu:", e)
    else:
        print("GPU kullanılmıyor. Dolayısıyla güç ölçümü geçersiz.")

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eğer Colab kullanıyorsanız, Drive bağlama işlemi:
from google.colab import drive
drive.mount('/content/drive')
data_dir = '/content/drive/MyDrive/giik/Eyes'

# Görüntü yollarını ve etiketlerini toplama
image_paths = []
labels = []

for label_dir in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, label_dir)
    if os.path.isdir(class_dir):
        for image_file in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, image_file))
            labels.append(label_dir)

# Sınıfları etiketlerle eşleştirme
class_names = list(set(labels))
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
labels = [class_to_idx[label] for label in labels]

# Eğitim ve test veri setlerine bölme
from sklearn.model_selection import train_test_split
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

def enhance_contrast(image):
    # Görüntüyü numpy array'e dönüştür
    image = np.array(image)

    # Eğer görüntü tek kanallı ise (grayscale)
    if len(image.shape) == 2:
        image = cv2.equalizeHist(image)
    else:  # RGB görüntüler için her kanalı ayrı eşitle
        y_cr_cb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y_cr_cb[:, :, 0] = cv2.equalizeHist(y_cr_cb[:, :, 0])
        image = cv2.cvtColor(y_cr_cb, cv2.COLOR_YCrCb2RGB)

    return Image.fromarray(image)

# Custom Dataset sınıfı
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    
# Veri ön işleme
transform = transforms.Compose([
    transforms.Lambda(lambda img: enhance_contrast(img)),  # Kontrast eşitleme
    transforms.Resize((256, 256)),  # Çözünürlüğü 256x256 olarak ayarla
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# DataLoader
train_dataset = CustomDataset(train_paths, train_labels, transform=transform)
test_dataset = CustomDataset(test_paths, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Model: CNN + LSTM + Attention
class CNN_LSTM_Attention(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM_Attention, self).__init__()

        # CNN katmanı
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),  # Dropout eklendi
        )

        # LSTM katmanı
        self.lstm = nn.LSTM(input_size=128 * 32 * 32, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(0.3)  # LSTM sonrası Dropout eklendi

        # Attention katmanı
        self.attention = nn.Linear(128 * 2, 1)

        # Fully Connected katman
        self.fc1 = nn.Linear(128 * 2, 256)  # Eklenen Dense katman
        self.fc2 = nn.Linear(256, num_classes)  # Final sınıflandırma katmanı
        self.dropout = nn.Dropout(0.3)  # Dropout eklenebilir

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.view(batch_size, -1, 128 * 32 * 32)

        # LSTM çıkışı
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)  # Dropout uygulanıyor

        # Attention hesaplama
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        #  Fully Connected katman
        x = self.fc1(context_vector)  # Eklenen Dense katman
        x = self.dropout(x)  # Dropout uygulanıyor
        out = self.fc2(x)  # Final sınıflandırma
        return out
# Model oluşturma
model = CNN_LSTM_Attention(num_classes=len(class_names)).to(device)

# Optimizasyon ve kayıp fonksiyonu
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2 düzenleme eklendi
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Eğitim ve test kayıplarını ve doğruluklarını izlemek için listeler
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# ---------------------------------
# EĞİTİM BAŞLANGICI
# ---------------------------------
epochs = 10

# Toplam eğitim süresini ölçmek için
train_start_time = time.time()

for epoch in range(epochs):
    # Epoch başlangıcında zaman ölçümü
    epoch_start_time = time.time()

    # Kaynak kullanım bilgileri
    print(f"\nEpoch {epoch+1}/{epochs} - Kaynak Kullanımı:")
    log_resource_usage()
    get_gpu_power_usage()

    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss /= len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Test seti üzerinde değerlendir
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # Öğrenme oranı adımı
    scheduler.step()

    # Epoch sonuçları
    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Bu epoch ne kadar sürdü?
    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch+1} süresi: {epoch_duration:.2f} saniye")

# Toplam eğitim süresi
train_end_time = time.time()
total_training_time = train_end_time - train_start_time
print(f"\nToplam Eğitim Süresi: {total_training_time:.2f} saniye")

# ---------------------------------
# Eğitim sonrası metrik görselleştirme
# ---------------------------------
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train vs Test Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train vs Test Accuracy')

plt.tight_layout()
plt.show()

# ---------------------------------
# Final değerlendirme (Test set)
# ---------------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Sınıflandırma raporu
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))