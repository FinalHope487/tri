import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from model import create_resnet34

# 自定義資料集
class ImageRegressionDataset(Dataset):
    def __init__(self, folder_path, transform=None, augmentation=False, k=2):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]
        self.augmentation = augmentation
        self.k = k

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        label_path = os.path.splitext(img_path)[0] + ".txt"

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r') as f:
            line = f.read().strip().replace(",", " ")
            label = torch.tensor([float(x) for x in line.split()], dtype=torch.float32)

        return image, label

# 圖像轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 參數設定
side = 'b'
main_folder = r"C:\Users\sword\.vscode\vtb\my-projects\tri"
dataset_folder = f'{main_folder}/dataset/{side}'
checkpoint_path = f"{main_folder}/trained_models/{side}/checkpoint.pt"
best_model_path = f"{main_folder}/trained_models/{side}/best.pt"
log_file = f"{main_folder}/trained_models/{side}/loss_log.txt"
batch_size = 8
learning_rate = 1e-4
num_epochs = 100
save_interval = 10

# 載入資料集並切分驗證集
full_dataset = ImageRegressionDataset(dataset_folder, transform)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 建立模型
model = create_resnet34()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 使用 GPU
device = torch.device("cuda")
model.to(device)

# 檢查是否有存檔可以續訓
start_epoch = 0
best_val_loss = float('inf')
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    print(f"✅ 已載入 checkpoint（從 epoch {start_epoch} 繼續）")

# 開始訓練
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # 驗證階段
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(val_loader)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 儲存 log
    with open(log_file, "a") as f:
        f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f}\n")

    # 儲存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"💾 儲存最佳模型至 {best_model_path}（val loss: {best_val_loss:.6f}）")

    # 每 N epoch 儲存 checkpoint
    if (epoch + 1) % save_interval == 0:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, checkpoint_path)
        print(f"📦 Checkpoint 儲存於 {checkpoint_path}")
