import pandas as pd

# 1 원본 파일 읽기
df = pd.read_csv("eyas_gpu4090_dataset_resnet50.csv")

# 2 뒤에서 16000개 추출
df_tail = df.tail(16000)

# 3 trimmed 파일 아래에 이어붙이기
df_tail.to_csv(
    "eyas_gpu4090_dataset_resnet50_trimmed.csv",
    mode='a',        # append 모드
    header=False,    # 컬럼명 중복 방지
    index=False
)