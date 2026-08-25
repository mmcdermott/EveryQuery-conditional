"""Does the cohort vocabulary actually contain TIMELINE//DELTA* codes?

If it does not, `strip_delta_tokens=true` strips nothing (the dataset logs a warning and still
emits `time_pos_ids`), so RoPE-time would be only half the feature.  Prints counts only.
"""

import os

import polars as pl

p = os.path.join(os.environ["TENSORIZED_COHORT_DIR"], "metadata", "codes.parquet")
c = pl.read_parquet(p).select("code")
n_delta = c.filter(pl.col("code").str.starts_with("TIMELINE//DELTA")).height
n_tl = c.filter(pl.col("code").str.starts_with("TIMELINE")).height
print(f"cohort codes           : {c.height}")
print(f"TIMELINE//DELTA* codes : {n_delta}   <- stripped when strip_delta_tokens=true")
print(f"TIMELINE* codes (any)  : {n_tl}")
print(f"rope-time is a REAL strip: {n_delta > 0}")
