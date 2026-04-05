import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# --- TVM 스타일의 하드웨어 설정 ---
# DEVICE: 실제 학습을 수행할 물리적 장치 (dev = tvm.cuda(0))
DEVICE = 'cuda:0' 
# TARGET: TVM 컴파일 타겟 설명 (RTX 4090은 sm_89 사용)
TARGET = "cuda -arch=sm_89"

def run_power_prediction():
    # 1. 파일 경로 설정
    file_path = os.path.expanduser('~/tvm/tutorials/eyas_gpu4090_dataset_resnet50.csv')
    
    if not os.path.exists(file_path):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path}")
        return

    print(f"[INFO] 데이터 로드 중: {file_path}")
    df = pd.read_csv(file_path)

    # 2. 특징(X)과 예측 목표(y) 분리
    # freq_mhz를 포함하여 학습을 진행합니다.
    drop_cols = ['i', 'model', 'workload_hash', 'trace_hash', 'avg_power_w', 'lat_mean_ms']
    X = df.drop(columns=drop_cols)
    y = df['avg_power_w']

    # 3. Stratified Split (전력 구간별 균등 분할)
    print("[INFO] Stratified Split을 위한 전력 구간 생성 중...")
    power_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')

    # 4. 데이터 분할 (8:1:1)
    indices = np.arange(len(df))
    idx_temp, idx_test, X_temp, X_test, y_temp, y_test, _, _ = train_test_split(
        indices, X, y, power_bins, test_size=0.1, stratify=power_bins, random_state=42
    )
    
    new_bins_temp = pd.qcut(y_temp, q=10, labels=False, duplicates='drop')
    idx_train, idx_val, X_train, X_val, y_train, y_val = train_test_split(
        idx_temp, X_temp, y_temp, test_size=0.1111, stratify=new_bins_temp, random_state=42
    )

    print(f"[INFO] 데이터 분할 완료 -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 5. XGBoost 모델 정의
    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device=DEVICE,
        early_stopping_rounds=50,
        eval_metric='mape',
        random_state=42
    )

    # 6. 학습 시작
    print(f"[INFO] 학습 시작 (Target: {TARGET}, Device: {DEVICE})")
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # 7. 성능 평가 및 오차율 출력
    # Validation 오차율은 학습 결과에서 가져옵니다.
    best_val_mape = model.best_score * 100
    
    # Test 오차율 계산
    y_pred = model.predict(X_test)
    test_mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*50)
    print(f"최종 모델 성능 리포트")
    print(f"-"*50)
    print(f"Best Validation MAPE : {best_val_mape:.4f}%")
    print(f"Final Test MAPE      : {test_mape:.4f}%")
    print(f"R-squared Score      : {r2:.4f}")
    print("="*50)

    # 8. 심층 분석 (Worst 10) - Frequency 포함
    abs_error = np.abs(y_test - y_pred)
    worst_idx = np.argsort(abs_error)[-10:][::-1]
    worst_orig_idx = idx_test[worst_idx]

    print("\n[DIAGNOSTICS] 오차가 가장 큰 상위 10개 데이터 분석 (주파수 포함):")
    print(f"{'순위':<4} | {'Freq(MHz)':<10} | {'Actual(W)':<10} | {'Pred(W)':<10} | {'MAPE(%)':<8} | {'Hash':<20}")
    print("-" * 100)
    
    for i, (orig_idx, pred) in enumerate(zip(worst_orig_idx, y_pred[worst_idx])):
        row = df.iloc[orig_idx]
        actual = row['avg_power_w']
        freq = row['freq_mhz']
        pct_err = (abs(actual - pred) / actual) * 100
        
        print(f"{i+1:<4} | {freq:<10.1f} | {actual:<10.1f} | {pred:<10.1f} | {pct_err:<8.2f} | {row['workload_hash']}")
    print("-" * 100)

    # 9. 모델 저장
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_save_path = os.path.join('models', 'power_model_sm_89.json')
    model.save_model(model_save_path)
    print(f"\n[INFO] 모델 저장 완료: {model_save_path}")

if __name__ == "__main__":
    run_power_prediction()