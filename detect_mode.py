import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import create_classifier

from config import SCREENSHOTS_FOLDER, get_best_model_path

dataset_folder = SCREENSHOTS_FOLDER
model_path = get_best_model_path('classifier')  # 可換成 checkpoint.pt

device = torch.device("cuda")

# # 載入checkpoint模型
# resnet18 = model.create_resnet18().to(device)
# checkpoint = torch.load(model_path, map_location=device)
# resnet18.load_state_dict(checkpoint["model_state"])
# resnet18.eval()

resnet18 = create_classifier().to(device)
best = torch.load(model_path, map_location=device)
resnet18.load_state_dict(best)
resnet18.eval()

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=dataset_folder, transform=val_transforms)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

print("開始模型檢測...\n")

correct = 0
total = 0

for img, label in tqdm(data_loader):
    img, label = img.to(device), label.to(device)
    output = resnet18(img)
    pred = torch.argmax(output, dim=1)

    true_class = idx_to_class[label.item()]
    pred_class = idx_to_class[pred.item()]
    result = "Pass" if pred.item() == label.item() else "Fail"

    img_path = dataset.samples[total][0]
    filename = os.path.basename(img_path)

    print(f"{filename:20} | True: {true_class:10} | Pred: {pred_class:10} | {result}")

    total += 1
    if pred.item() == label.item():
        correct += 1

acc = correct / total
print(f"\n預測準確率：{correct}/{total} ({acc*100:.2f}%)")
