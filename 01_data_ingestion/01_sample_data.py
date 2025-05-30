# %%
"""
Script to sample a small subset of the fake news data for demonstration purposes.
"""

# %%
import pandas as pd
import os
import random

# %%
# Set random seed for reproducibility
random.seed(42)

# %%
# Define paths
input_dir = "/home/ubuntu/upload"
output_dir = "/home/ubuntu/fake_news_detection/data"

# %%
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# %%
# Define sample size
sample_size = 100  # 100 samples from each class

# %%
# Read and sample fake news data
fake_df = pd.read_csv(f"{input_dir}/Fake.csv")
fake_sample = fake_df.sample(n=min(sample_size, len(fake_df)), random_state=42)
fake_sample['label'] = 0  # 0 for fake news

# %%
# Read and sample true news data
true_df = pd.read_csv(f"{input_dir}/True.csv")
true_sample = true_df.sample(n=min(sample_size, len(true_df)), random_state=42)
true_sample['label'] = 1  # 1 for true news

# %%
# Combine samples
combined_sample = pd.concat([fake_sample, true_sample], ignore_index=True)

# %%
# Shuffle the combined sample
combined_sample = combined_sample.sample(frac=1, random_state=42).reset_index(drop=True)

# %%
# Save the combined sample
combined_sample.to_csv(f"{output_dir}/news_sample.csv", index=False)

# %%
# Create a small streaming sample
stream_sample = combined_sample.sample(n=20, random_state=42)
stream_sample['id'] = [f"doc_{i}" for i in range(len(stream_sample))]
stream_sample['timestamp'] = pd.Timestamp.now()
stream_sample.to_csv(f"{output_dir}/stream_sample.csv", index=False)

# %%
print(f"Created sample dataset with {len(combined_sample)} records")
print(f"Created streaming sample with {len(stream_sample)} records")
print(f"Files saved to {output_dir}")
