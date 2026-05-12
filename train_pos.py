import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from model import create_player

class ImageRegressionDataset(Dataset):
    def __init__(self, folder_path, transform=None, augmentation=False, k=2):
        self.folder_path = folder_path
        self.transform = transform
        self.augmentation = augmentation
        self.k = k

        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]
        
        if self.augmentation:
            self.image_aug_pairs = []
            for f in self.image_files:
                for i in range(k + 1):  # 原圖 + k 次增強
                    self.image_aug_pairs.append((f, i))  # i=0 表示原圖，i>0 表示增強版本
        else:
            self.image_aug_pairs = [(f, 0) for f in self.image_files]

    def __len__(self):
        return len(self.image_aug_pairs)

    def __getitem__(self, idx):
        img_name, aug_idx = self.image_aug_pairs[idx]
        img_path = os.path.join(self.folder_path, img_name)
        label_path = os.path.splitext(img_path)[0] + ".txt"

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        # 讀取 label (x_norm, y_norm)
        with open(label_path, 'r') as f:
            line = f.read().strip().replace(",", " ")
            x_norm, y_norm = map(float, line.split())
            x_pix = x_norm * W
            y_pix = y_norm * H

        if aug_idx > 0 and self.augmentation:
            image, (x_pix, y_pix) = self._augment_image(image, x_pix, y_pix)

        # 轉換為正規化座標
        x_norm_new = x_pix / W
        y_norm_new = y_pix / H
        label = torch.tensor([x_norm_new, y_norm_new], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

    def _augment_image(self, image, x, y):
        while True:
            W, H = image.size
            scale = random.uniform(0.8, 1.2)
            dx = random.uniform(-0.1, 0.1) * W
            dy = random.uniform(-0.1, 0.1) * H

            matrix = (
                scale, 0, dx,
                0, scale, dy
            )

            image_aug = image.transform((W, H), Image.AFFINE, matrix, resample=Image.BILINEAR)

            # 調整點位
            x_new = x / scale - dx
            y_new = y / scale - dy

            if abs(x_new / W - 0.5) < 0.5 or abs(y_new / H - 0.5) < 0.5: # 避免數值超出範圍
                break

        return image_aug, (x_new, y_new)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def main():
    for side in ['a', 'b', 'c']:
        from config import DATASET_FOLDER, get_checkpoint_path, get_best_model_path, get_log_file_path
        print(f"Training {side} side model ...")
        # side = 'c'
        dataset_folder = str(DATASET_FOLDER / side)
        checkpoint_path = str(get_checkpoint_path(side))
        best_model_path = str(get_best_model_path(side))
        log_file = str(get_log_file_path(side))
        batch_size = 8
        learning_rate = 1e-4
        num_epochs = 200
        save_interval = 10

        full_dataset = ImageRegressionDataset(dataset_folder, transform, augmentation=False, k=2)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print('資料集載入已完成')

        model = create_player()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        device = torch.device("cuda")
        model.to(device)

        start_epoch = 0
        best_val_loss = float('inf')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            print(f"已載入 checkpoint（從 epoch {start_epoch} 繼續）")

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

            with open(log_file, "a") as f:
                f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f}\n")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"儲存最佳模型至 {best_model_path}（val loss: {best_val_loss:.6f}）")

            if (epoch + 1) % save_interval == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss
                }, checkpoint_path)
                print(f"Checkpoint 儲存於 {checkpoint_path}")
            
        print(f"Training {side} side model completed")
        print("=====================================")

if __name__ == "__main__":
    main()