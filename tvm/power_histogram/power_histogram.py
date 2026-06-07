import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/home/hyunjae/tvm/tutorials/eyas_gpu4090_dataset_resnet50.csv"

df = pd.read_csv(csv_path)

# ✅ frequency 필터링
filtered_df = df[df["freq_mhz"].isin([2235, 2520])]

# ❗ 데이터 확인 (중요)
print(filtered_df["freq_mhz"].value_counts())

power = filtered_df["avg_power_w"]

plt.figure()

plt.hist(power, bins=50)

plt.xlabel("Power (W)")
plt.ylabel("Count")
plt.title("Power Distribution (2235 & 2520 MHz)")

plt.savefig("power_histogram_filtered.png")

print("Saved to power_histogram_filtered.png")