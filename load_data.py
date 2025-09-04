import os
import pandas as pd
data_dir = "data"

"""Load files from the local directory"""
stocks_df = pd.read_parquet(os.path.join(data_dir, "stocks_df.parquet"))
econ_indices_df = pd.read_parquet(os.path.join(data_dir, "econ_indices_df.parquet"))
macros_df = pd.read_parquet(os.path.join(data_dir, "macros_df.parquet"))
