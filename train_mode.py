import os
import torch
from torchvision import transforms
from PIL import Image
import pygame

from model import create_resnet34

# ---------- 設定 ----------
side = 'a'
main_folder = r"C:\Users\sword\.vscode\vtb\my-projects\tri"
dataset_folder = f"{main_folder}/dataset/{side}"
best_model_path = f"{main_folder}/trained_models/{side}/best.pt"

# ---------- 載入模型 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_resnet34()
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()

# ---------- 圖像處理 ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------- 讀取所有圖片 ----------
image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith((".jpg", ".png"))]
image_files.sort()  # 若檔名有順序需求

results = []

# ---------- 批次預測 ----------
for img_file in image_files:
    img_path = os.path.join(dataset_folder, img_file)
    label_path = os.path.splitext(img_path)[0] + ".txt"

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor).squeeze().cpu()  # tensor([x, y])

    pred_x, pred_y = output.tolist()

    # Ground truth
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            gt = [float(x) for x in f.read().strip().replace(",", " ").split()]
            gt_x, gt_y = gt
    else:
        gt_x, gt_y = None, None

    results.append({
        "filename": img_file,
        "image": image,
        "pred": (pred_x, pred_y),
        "gt": (gt_x, gt_y)
    })

# ---------- 初始化 Pygame ----------
pygame.init()
display_size = (448, 448)
screen = pygame.display.set_mode(display_size)
pygame.display.set_caption("🔍 模型推論結果（← / → 切換）")
font = pygame.font.SysFont(None, 24)

# ---------- 顯示邏輯 ----------
index = 0
running = True

def draw_result(result):
    screen.fill((0, 0, 0))
    image_resized = result["image"].resize(display_size)
    img_data = pygame.image.fromstring(image_resized.tobytes(), image_resized.size, image_resized.mode)
    screen.blit(img_data, (0, 0))

    # 畫預測點（紅）
    px = int(result["pred"][0] * display_size[0])
    py = int(result["pred"][1] * display_size[1])
    pygame.draw.circle(screen, (255, 0, 0), (px, py), 5)

    # 畫真實點（綠）
    if result["gt"][0] is not None:
        gx = int(result["gt"][0] * display_size[0])
        gy = int(result["gt"][1] * display_size[1])
        pygame.draw.circle(screen, (0, 255, 0), (gx, gy), 5)

    # 顯示圖片名稱
    label_surface = font.render(result["filename"], True, (255, 255, 255))
    screen.blit(label_surface, (10, 10))

    pygame.display.flip()

# ---------- 主迴圈 ----------
draw_result(results[index])

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                index = (index + 1) % len(results)
                draw_result(results[index])
            elif event.key == pygame.K_LEFT:
                index = (index - 1) % len(results)
                draw_result(results[index])

pygame.quit()

