from pathlib import Path

# 基礎目錄 (自動取得 config.py 所在的目錄)
MAIN_FOLDER = Path(__file__).parent

# 各子目錄路徑
DATASET_FOLDER = MAIN_FOLDER / "dataset"
DATASET_FULLSCREEN_FOLDER = MAIN_FOLDER / "dataset(FULL_SCREEN)"
TRAINED_MODELS_FOLDER = MAIN_FOLDER / "trained_models"
SCREENSHOTS_FOLDER = MAIN_FOLDER / "screenshots"

# 取得特定 side 的模型路徑
def get_best_model_path(side):
    if side == "classifier":
        return TRAINED_MODELS_FOLDER / "classifier" / "best.pt"
    return TRAINED_MODELS_FOLDER / side / "best.pt"

def get_checkpoint_path(side):
    if side == "classifier":
        return TRAINED_MODELS_FOLDER / "classifier" / "checkpoint.pt"
    return TRAINED_MODELS_FOLDER / side / "checkpoint.pt"

def get_log_file_path(side):
    if side == "classifier":
        return TRAINED_MODELS_FOLDER / "classifier" / "loss_log.txt"
    return TRAINED_MODELS_FOLDER / side / "loss_log.txt"
