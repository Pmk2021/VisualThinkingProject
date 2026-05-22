# https://docs.ultralytics.com/guides/finetuning-guide#basic-fine-tuning-example

# /data/images/train/image1.jpg
# /data/images/train/image2.jpg
# /data/images/val/image1.jpg
# /data/images/val/image2.jpg

# /data/labels/train/image1.txt
# /data/labels/train/image2.txt
# /data/labels/val/image1.txt
# /data/labels/val/image2.txt

# class_id center_x center_y width height

from ultralytics import YOLO
from pathlib import Path
import platform
import re

# Find the config

NODE = platform.node()
NODE_TEMPLATE = r"\b(izar1|i[0-9]{2}|ixl[0-9]{2})\b"
IZAR = re.fullmatch(NODE_TEMPLATE, NODE) is not None

DATA_ROOT = Path("/work/cs-503/gromb/waymoyolo") if IZAR else Path("/Users/nathangromb/Documents/MA4/VI/project/data/waymoyolo")
CONFIG_PATH = DATA_ROOT / "data.yaml"

model = YOLO("yolo26n.pt")  # load pretrained model
model.train(
    data=str(CONFIG_PATH),
    epochs=10,
    imgsz=640, 
    freeze=10,
    patience=5,
    batch=-1,
    save_period=1,
    device=0 if IZAR else "cpu",
    seed=42,
)

model_path = "/work/cs-503/gromb/waymoyolo/yolo_finetuned.pt" if IZAR else Path(__file__).parent / "yolo_finetuned.pt"
model.save(model_path)