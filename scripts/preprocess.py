import os
import pandas as pd
import re
from sklearn.utils import resample

# -----------------------
# Automatically find project root and set folders
# -----------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_path = os.path.join(project_root, "data", "raw")
processed_path = os.path.join(project_root, "data", "processed")
os.makedirs(processed_path, exist_ok=True)

# -----------------------
# Helper function to clean text
# -----------------------
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)                      # remove mentions
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)          # remove special chars
    text = re.sub(r"\s+", " ", text).strip()             # remove extra spaces
    return text

# -----------------------
# Load datasets
# -----------------------
datasets = {
    "hate_speech": ["hate_speech.csv", "hate_speech_train.csv", "hate_speech_test.csv"],
    "suicide": ["Suicide_Detection.csv"],
    "extremism": ["extremism_data_final.csv"],
    "fake_news": ["fake_or_real_news.csv"]
}

all_data = []

for label, files in datasets.items():
    temp = []
    for file in files:
        path = os.path.join(raw_path, file)
        if not os.path.exists(path):
            print(f"Warning: File not found -> {path}")
            continue

        # Safe CSV reading with fallback encoding
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin1")

        # detect text column
        if "text" in df.columns:
            texts = df["text"]
        elif "tweet" in df.columns:
            texts = df["tweet"]
        else:
            texts = df.iloc[:,0]

        df_clean = pd.DataFrame({
            "text": texts.astype(str).apply(clean_text),
            "label": label
        })
        temp.append(df_clean)

    if temp:
        all_data.append(pd.concat(temp, ignore_index=True))

# -----------------------
# Balance datasets by smallest class
# -----------------------
min_len = min(len(df) for df in all_data)
balanced_data = [resample(df, replace=False, n_samples=min_len, random_state=42) for df in all_data]

# -----------------------
# Combine all datasets
# -----------------------
final_df = pd.concat(balanced_data, ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# -----------------------
# Save final dataset
# -----------------------
final_csv = os.path.join(processed_path, "final_dataset.csv")
final_df.to_csv(final_csv, index=False, encoding="utf-8")

print(f"✅ Final dataset saved at: {final_csv}")
print("Class distribution:")
print(final_df["label"].value_counts())