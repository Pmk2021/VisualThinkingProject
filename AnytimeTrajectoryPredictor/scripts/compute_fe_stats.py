import platform
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import argparse
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
import torch

NODE = platform.node()
NODE_TEMPLATE = r"\b(izar1|i[0-9]{2}|ixl[0-9]{2})\b"
IZAR = re.fullmatch(NODE_TEMPLATE, NODE) is not None

GT_PATH = Path("/work/cs-503/santanto/waymo") if IZAR else Path("/Users/nathangromb/Documents/MA4/VI/project/data")
FE_PATH = Path("/work/cs-503/gromb/waymo") if IZAR else Path("/Users/nathangromb/Documents/MA4/VI/project/data")
OUT_PATH = Path("/work/cs-503/gromb/fe_stats") if IZAR else Path("/Users/nathangromb/Documents/MA4/VI/project/data/fe_stats")

GT_FILENAME = "image_trajectories.parquet"
FE_FILENAME = "fe_bboxes.parquet"
CONF_FILENAME = "fe_confidences.parquet"
CLASS_ID_FILENAME = "fe_class_ids.parquet"

if IZAR:
    DIRS = list(Path(GT_PATH).glob("training__*")) + list(Path(GT_PATH).glob("validation__*"))
    DIRS = [d.name for d in DIRS]
else:
    DIRS = list(Path(GT_PATH).glob("waymo"))
    DIRS = [d.name for d in DIRS]

YOLO_TO_WAYMO = {
    0: 2,   # person        -> pedestrian
    1: 4,   # bicycle       -> cyclist
    2: 1,   # car           -> vehicle
    3: 4,   # motorcycle    -> cyclist
    5: 1,   # bus           -> vehicle
    6: 1,   # train         -> vehicle
    7: 1,   # truck         -> vehicle
    9: 3,   # traffic light -> sign
    11: 3,  # stop sign     -> sign
}

def cxcywh_to_xyxy(boxes_cxcywh: list[tuple]) -> torch.Tensor:
    """Convert list of (cx, cy, w, h) to tensor of (x1, y1, x2, y2) safely."""
    if not boxes_cxcywh:
        return torch.empty((0, 4), dtype=torch.float32)
    
    t = torch.tensor(boxes_cxcywh, dtype=torch.float32)
    return torch.stack([
        t[:, 0] - t[:, 2] / 2,  # x1
        t[:, 1] - t[:, 3] / 2,  # y1
        t[:, 0] + t[:, 2] / 2,  # x2
        t[:, 1] + t[:, 3] / 2,  # y2
    ], dim=1)

def process_image_geometry(gt_bboxes: list[tuple], fe_bboxes: list[tuple]):
    """Extracts tensors and computes image-level metric components."""
    if not gt_bboxes or not fe_bboxes:
        # Return zeros for IoU tracking if there are no targets or predictions
        return None, None, None, None, None, 0.0, torch.tensor([])

    gt_boxes   = cxcywh_to_xyxy([b[:4] for b in gt_bboxes])
    pred_boxes = cxcywh_to_xyxy([b[:4] for b in fe_bboxes])
    
    scores     = torch.tensor([b[4] for b in fe_bboxes], dtype=torch.float32)
    labels     = torch.tensor([YOLO_TO_WAYMO.get(b[5], 0) for b in fe_bboxes], dtype=torch.int)
    gt_labels  = torch.tensor([b[4] for b in gt_bboxes], dtype=torch.int)

    iou_matrix = box_iou(gt_boxes, pred_boxes)
    
    # Best IoU matching score for each individual Ground Truth object box
    gt_best_ious = iou_matrix.max(dim=1).values if iou_matrix.numel() > 0 else torch.zeros(len(gt_boxes))
    mean_iou = gt_best_ious.mean().item()

    return gt_boxes, pred_boxes, scores, labels, gt_labels, mean_iou, gt_best_ious

def process_dir(dir_name, force=False, verbose=False):
    out_file = OUT_PATH / "chunks" / f"{dir_name}_stats.json"
    if not force and out_file.exists():
        print(f"Statistics for {dir_name} already exist, skipping...")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)

    join_cols = ['split', 'image_id', 'scene_id', 'frame_timestamp_micros', 'camera_name', 'camera_name_text']
    
    # Load Parquet Data
    gt_df = pd.read_parquet(GT_PATH / dir_name / GT_FILENAME)
    fe_df = pd.read_parquet(FE_PATH / dir_name / FE_FILENAME)
    conf_df = pd.read_parquet(FE_PATH / dir_name / CONF_FILENAME)
    class_id_df = pd.read_parquet(FE_PATH / dir_name / CLASS_ID_FILENAME)

    if not fe_df.shape[0] == conf_df.shape[0]:
        raise ValueError("Feature extractor rows mismatch with confidence rows.")
    
    fe_df = fe_df.merge(conf_df, on=join_cols + ['object_id'], how='left')
    fe_df = fe_df.merge(class_id_df, on=join_cols + ['object_id'], how='left')
    
    if fe_df['class_id'].isna().sum() > 0:
        raise ValueError(f"Missing class_id values after merge: {fe_df['class_id'].isna().sum()}")

    # Vectorized fast column mapping
    gt_df['bbox'] = list(zip(gt_df['bbox_center_x'], gt_df['bbox_center_y'], gt_df['bbox_width'], gt_df['bbox_height'], gt_df['object_type']))
    fe_df['bbox'] = list(zip(fe_df['cx'], fe_df['cy'], fe_df['w'], fe_df['h'], fe_df['confidence'], fe_df['class_id']))

    gt_df_grouped = gt_df[join_cols + ['bbox']].groupby(join_cols).agg(list).reset_index()
    fe_df_grouped = fe_df[join_cols + ['bbox']].groupby(join_cols).agg(list).reset_index()

    joined = pd.merge(gt_df_grouped, fe_df_grouped, on=join_cols, how='inner', suffixes=('_gt', '_fe'))

    bboxs_gt = {tuple(row[join_cols]): row['bbox_gt'] for _, row in joined.iterrows()}
    bboxs_fe = {tuple(row[join_cols]): row['bbox_fe'] for _, row in joined.iterrows()}

    if len(bboxs_gt) != len(bboxs_fe):
        raise ValueError(f"Unique image mismatch between GT and FE: {len(bboxs_gt)} vs {len(bboxs_fe)}")

    # --- Initialize Metrics and IoU Aggregators ---
    chunk_metric = MeanAveragePrecision(iou_type="bbox")
    image_stats_summary = {}
    valid_images_processed = 0
    
    total_image_iou_sum = 0.0
    all_gt_object_ious = []

    for image_id, gt_bboxes in bboxs_gt.items():
        fe_bboxes = bboxs_fe.get(image_id, [])
        
        gt_boxes, pred_boxes, scores, labels, gt_labels, mean_iou, gt_best_ious = process_image_geometry(gt_bboxes, fe_bboxes)
        
        # Accumulate metrics
        image_stats_summary[str(image_id)] = {"mean_iou": mean_iou}
        total_image_iou_sum += mean_iou
        
        if gt_best_ious.numel() > 0:
            all_gt_object_ious.append(gt_best_ious)
        else:
            # If an image has GT boxes but 0 predictions, their individual best match IoU is 0.0
            all_gt_object_ious.append(torch.zeros(len(gt_bboxes)))

        if gt_boxes is not None and pred_boxes is not None and len(gt_boxes) > 0 and len(pred_boxes) > 0:
            chunk_metric.update(
                preds=[{"boxes": pred_boxes, "scores": scores, "labels": labels}],
                target=[{"boxes": gt_boxes, "labels": gt_labels}],
            )
            valid_images_processed += 1

    # --- Finalize Chunk-Level Metric Calculations ---
    if valid_images_processed > 0:
        map_result = chunk_metric.compute()
        chunk_map_50 = map_result["map_50"].item()
        chunk_map_50_95 = map_result["map"].item()
    else:
        chunk_map_50, chunk_map_50_95 = 0.0, 0.0

    # Compute Global Chunk IoU averages
    num_images = len(bboxs_gt)
    chunk_mean_iou_per_image = total_image_iou_sum / num_images if num_images > 0 else 0.0
    
    if all_gt_object_ious:
        chunk_mean_iou_per_object = torch.cat(all_gt_object_ious).mean().item()
    else:
        chunk_mean_iou_per_object = 0.0

    # Build final chunk payload
    stats = {
        "num_unique_images": num_images,
        "num_gt_bboxes": sum(len(bboxes) for bboxes in bboxs_gt.values()),
        "num_fe_bboxes": sum(len(bboxes) for bboxes in bboxs_fe.values()),
        "map_50": chunk_map_50,
        "map_50_95": chunk_map_50_95,
        "mean_iou_per_image": chunk_mean_iou_per_image,   # Average score across all frames
        "mean_iou_per_object": chunk_mean_iou_per_object, # Average score across all unique objects
    }
    
    json.dump(stats, out_file.open("w"), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute feature extractor statistics chunk-wise with global IoU tracking")
    parser.add_argument("--force", action="store_true", help="Force recomputation of statistics")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information during processing")
    args = parser.parse_args()

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    for dir_name in tqdm(DIRS, desc="Processing directories"):
        process_dir(dir_name, force=args.force, verbose=args.verbose)