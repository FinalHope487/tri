import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import create_classifier

# ========== 1. 設定路徑 ==========
main_folder = r"C:\Users\sword\.vscode\vtb\my-projects\tri"
dataset_folder = f"{main_folder}/screenshots"
checkpoint_path = f"{main_folder}/trained_models/classifier/checkpoint.pt"
best_model_path = f"{main_folder}/trained_models/classifier/best.pt"
log_file = f"{main_folder}/trained_models/classifier/loss_log.txt"
final_model_path = f"{main_folder}/trained_models/classifier/model_classifier.pt"

os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# ========== 2. 設定參數 ==========
num_classes = 3
batch_size = 32
num_epochs = 50
learning_rate = 1e-4
device = torch.device("cuda")

# ========== 3. 資料轉換 ==========
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
    transforms.RandomRotation(degrees=60),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.5)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========== 4. 載入資料集並切割訓練 / 驗證 ==========
full_dataset = datasets.ImageFolder(root=dataset_folder)
train_len = int(0.8 * len(full_dataset))
val_len = len(full_dataset) - train_len
train_set, val_set = random_split(full_dataset, [train_len, val_len])

# train_set.dataset.transform = train_transforms
# val_set.dataset.transform = val_transforms

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# ========== 5. 模型與訓練設定 ==========
model = create_classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_epoch = 0
best_val_loss = float("inf")

# 嘗試從 checkpoint 或 best.pt 恢復模型
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    print(f"🔄 從 checkpoint 回復訓練，繼續從 epoch {start_epoch} 開始")
    
elif os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    best_val_loss = checkpoint["best_val_loss"]
    print("📥 載入 best.pt 欲作為初始化訓練")

# 初始化 log 檔案（如不存在）
if not os.path.exists(log_file) or start_epoch == 0:
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

# ========== 6. 開始訓練 ==========
print("📦 開始訓練 model...\n")

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()

    train_loss = running_loss / len(train_set)
    train_acc = correct / len(train_set)

    # ========== 驗證階段 ==========
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()

    val_loss = val_loss / len(val_set)
    val_acc = val_correct / len(val_set)

    # ========== 顯示當前結果 ==========
    print(f"📊 Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    # ========== 儲存 log ==========
    with open(log_file, "a") as f:
        f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")

    # ========== 儲存最佳模型 ==========
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, best_model_path)
        print(f"💾 已儲存最佳模型 → best.pt (val_loss={best_val_loss:.4f})")

    # ========== 儲存 Checkpoint ==========
    if (epoch + 1) % 10 == 0:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, checkpoint_path)
        print(f"🕹️ Checkpoint 儲存：{checkpoint_path}")

# ========== 7. 儲存最終模型 ==========
torch.save(model.state_dict(), final_model_path)
print(f"\n✅ 最終模型儲存至：{final_model_path}")
