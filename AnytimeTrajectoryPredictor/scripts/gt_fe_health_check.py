"""Check consistency between GT feature parquet outputs and source tables.

Usage:
  python scripts/gt_fe_check.py --base data/waymo
"""
from pathlib import Path
import argparse
import sys
import pandas as pd


def find_fe_files(base: Path):
    pairs = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        local_fp = child / "fe_gt_local_latent_features.parquet"
        if not local_fp.exists():
            continue
        global_fp = child / "fe_gt_latent_features.parquet"
        pairs.append((child, local_fp, global_fp if global_fp.exists() else None))
    return pairs


def check_dir(dirpath: Path, local_fp: Path, global_fp: Path):
    print(f"Checking {dirpath}")
    issues = []
    images_fp = dirpath / "images.parquet"
    traj_fp = dirpath / "image_trajectories.parquet"
    if not images_fp.exists():
        msg = f"MISSING images.parquet at {images_fp}"
        print(f"  {msg}")
        return False, [msg]
    if not traj_fp.exists():
        msg = f"MISSING image_trajectories.parquet at {traj_fp}"
        print(f"  {msg}")
        return False, [msg]

    images = pd.read_parquet(images_fp)
    traj = pd.read_parquet(traj_fp)
    fe_local = pd.read_parquet(local_fp)
    fe_global = pd.read_parquet(global_fp) if global_fp is not None and global_fp.exists() else None

    print(fe_local.loc[:, list(fe_local.columns)[:12]].head())
    if fe_global is not None:
        print(fe_global.loc[:, list(fe_global.columns)[:7]].head())

    ok = True

    # Check image_id membership
    imgs_set = set(images["image_id"].astype(str))
    local_img_ids = set(fe_local["image_id"].astype(str))
    missing_imgs = local_img_ids - imgs_set
    if missing_imgs:
        msg = f"ERROR: {len(missing_imgs)} local features refer to missing image_ids (showing up to 5): {list(missing_imgs)[:5]}"
        print(f"  {msg}")
        issues.append(msg)
        ok = False

    # Check trajectory_id preservation
    if "trajectory_id" not in fe_local.columns:
        msg = "ERROR: fe_gt_local_latent_features.parquet missing 'trajectory_id' column"
        print(f"  {msg}")
        issues.append(msg)
        ok = False
    else:
        traj_ids = set(traj["trajectory_id"].astype(str))
        missing_tids = set(fe_local["trajectory_id"].astype(str)) - traj_ids
        if missing_tids:
            msg = f"ERROR: {len(missing_tids)} local features refer to unknown trajectory_id (showing up to 5): {list(missing_tids)[:5]}"
            print(f"  {msg}")
            issues.append(msg)
            ok = False

    # Per-image counts
    gt_counts = traj.groupby("image_id").size().to_dict()
    fe_counts = fe_local.groupby("image_id").size().to_dict()
    mismatches = []
    for img_id, gt_c in gt_counts.items():
        fe_c = fe_counts.get(img_id, 0)
        if gt_c != fe_c:
            mismatches.append((img_id, gt_c, fe_c))
            if len(mismatches) > 20:
                break
    if mismatches:
        msg = f"WARN: {len(mismatches)} images with mismatched object counts (showing up to 20)"
        print(f"  {msg}")
        issues.append(msg)
        for img_id, gt_c, fe_c in mismatches[:20]:
            print(f"    image_id={img_id}: gt={gt_c} fe_local={fe_c}")
        ok = False

    # Global features are optional in GT mode; skip checks if absent.
    if fe_global is not None:
        if "image_id" not in fe_global.columns:
            msg = "ERROR: fe_gt_latent_features.parquet missing 'image_id' column"
            print(f"  {msg}")
            issues.append(msg)
            ok = False
        else:
            missing_global = set(fe_global["image_id"].astype(str)) - imgs_set
            if missing_global:
                msg = f"ERROR: {len(missing_global)} global feature rows reference missing images"
                print(f"  {msg}")
                issues.append(msg)
                ok = False

    # Feature column presence (sanity)
    local_feat_cols = [c for c in fe_local.columns if c.startswith("local_latent_features_")]
    global_feat_cols = [c for c in fe_global.columns if c.startswith("latent_features_")] if fe_global is not None else []
    if not local_feat_cols:
        print("  WARN: no columns starting with 'local_latent_features_' found in local features")
    if not global_feat_cols:
        print("  WARN: no columns starting with 'latent_features_' found in global features")

    # Check that there are no all 0 feature rows (sanity)
    local_zero_rows = fe_local[local_feat_cols].apply(lambda row: (row == 0).all(), axis=1)
    if local_zero_rows.any():
        msg = f"WARN: {local_zero_rows.sum()} local feature rows are all zeros"
        print(f"  {msg}")
        issues.append(msg)
    if fe_global is not None and global_feat_cols:
        global_zero_rows = fe_global[global_feat_cols].apply(lambda row: (row == 0).all(), axis=1)
        if global_zero_rows.any():
            msg = f"WARN: {global_zero_rows.sum()} global feature rows are all zeros"
            print(f"  {msg}")
            issues.append(msg)

    if ok:
        print("  OK: checks passed for this directory")
    return ok, issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=str,
        default="/work/cs-503/gromb/waymo",
        help="Base path containing Waymo split folders",
    )
    args = parser.parse_args()
    base = Path(args.base)
    if not base.exists():
        print(f"Base path {base} not found")
        sys.exit(2)

    pairs = find_fe_files(base)
    if not pairs:
        print(f"No fe_gt local files found under {base}")
        local_only = [child / "fe_gt_local_latent_features.parquet" for child in sorted(base.iterdir()) if child.is_dir() and (child / "fe_gt_local_latent_features.parquet").exists()]
        global_only = [child / "fe_gt_latent_features.parquet" for child in sorted(base.iterdir()) if child.is_dir() and (child / "fe_gt_latent_features.parquet").exists()]
        print(f"Found local files: {len(local_only)}")
        print(f"Found global files: {len(global_only)}")
        if local_only:
            print("Example local file:", local_only[0])
        if global_only:
            print("Example global file:", global_only[0])
        sys.exit(1)

    all_ok = True
    failed = []
    for dirpath, local_fp, global_fp in pairs:
        ok, issues = check_dir(dirpath, local_fp, global_fp)
        all_ok = all_ok and ok
        if not ok:
            failed.append((dirpath, issues))

    if not all_ok:
        print("\nSummary of failures:")
        for dirpath, issues in failed:
            print(f"- {dirpath}")
            if issues:
                for issue in issues:
                    print(f"    * {issue}")
            else:
                print("    * Failed with no recorded issue details")
        print("One or more checks failed")
        sys.exit(1)
    print("All checks passed")


if __name__ == "__main__":
    main()
