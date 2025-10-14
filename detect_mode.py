import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import create_classifier

# ========== 設定路徑 ==========
main_folder = r"C:\Users\sword\.vscode\vtb\my-projects\tri"
dataset_folder = f"{main_folder}/screenshots"
model_path = f"{main_folder}/trained_models/classifier/best.pt"  # 可換成 checkpoint.pt

# ========== 裝置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========== 載入checkpoint模型 ==========
# resnet18 = model.create_resnet18().to(device)
# checkpoint = torch.load(model_path, map_location=device)
# resnet18.load_state_dict(checkpoint["model_state"])
# resnet18.eval()

# ========== 載入best模型 ==========
resnet18 = create_classifier().to(device)
best = torch.load(model_path, map_location=device)
resnet18.load_state_dict(best)
resnet18.eval()

# ========== 資料轉換 ==========
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========== 載入驗證資料集 ==========
dataset = datasets.ImageFolder(root=dataset_folder, transform=val_transforms)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 取得 class 對應關係
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

# ========== 開始檢測 ==========
print("🔍 開始模型檢測...\n")

correct = 0
total = 0

for img, label in tqdm(data_loader):
    img, label = img.to(device), label.to(device)
    output = resnet18(img)
    pred = torch.argmax(output, dim=1)

    true_class = idx_to_class[label.item()]
    pred_class = idx_to_class[pred.item()]
    result = "✅" if pred.item() == label.item() else "❌"

    # 抓取圖片的檔名
    img_path = dataset.samples[total][0]
    filename = os.path.basename(img_path)

    print(f"{filename:20} | True: {true_class:10} | Pred: {pred_class:10} | {result}")

    total += 1
    if pred.item() == label.item():
        correct += 1

# ========== 顯示總結 ==========
acc = correct / total
print(f"\n🎯 預測準確率：{correct}/{total} ({acc*100:.2f}%)")
