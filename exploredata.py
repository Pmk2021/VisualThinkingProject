import pandas as pd

# Path to your parquet file
parquet_path = "/work/cs-503/gromb/waymo/training__3194871563717679715_4980_000_5000_000/fe_gt_local_latent_features.parquet"

# Load the parquet file
df = pd.read_parquet(parquet_path)

# Print column names
print(df.columns.tolist())