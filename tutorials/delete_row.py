import pandas as pd

df = pd.read_csv("eyas_gpu4090_dataset_resnet50.csv")

# 뒤에서 8000개 제거
df = df.iloc[:-4633]

df.to_csv("eyas_gpu4090_dataset_resnet50_trimmed.csv", index=False)