import pandas as pd
import os

# The 10 basenames you specified to exclude
exclude_basenames = [
    "BM276:MV-00004:zedx:20250131-021847_0.jpg", "BM276:MV-00004:zedx:20250131-021904_70.jpg",
    "BM276:MV-00004:zedx:20250131-021904_90.jpg", "BM276:MV-00004:zedx:20250131-021921_132.jpg",
    "BM276:MV-00004:zedx:20250131-021921_4.jpg", "BM276:MV-00004:zedx:20250131-021921_51.jpg",
    "BM276:MV-00004:zedx:20250131-021921_72.jpg", "BM276:MV-00004:zedx:20250131-021921_73.jpg",
    "BM276:MV-00004:zedx:20250131-021921_8.jpg", "BM276:MV-00004:zedx:20250131-021847_43.jpg"
]

# Path to your master file
MASTER_CSV = 'data/20261023-captions_BM276(in).csv' 

if not os.path.exists(MASTER_CSV):
    print(f"Error: Could not find {MASTER_CSV}")
else:
    df = pd.read_csv(MASTER_CSV)
    # Filter out the 10 samples
    df_filtered = df[~df['image_file_path'].isin(exclude_basenames)]
    
    # Pick 100 random samples
    df_next_100 = df_filtered.sample(n=100, random_state=42)[['image_file_path', 'caption']]

    os.makedirs('data', exist_ok=True)
    df_next_100.to_csv('data/MMU_next_100_samples.csv', index=False)
    print(f"Success! Saved 100 new samples to data/MMU_next_100_samples.csv")