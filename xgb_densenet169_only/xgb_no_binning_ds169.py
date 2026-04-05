import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# --- TVM 스타일의 하드웨어 설정 ---
DEVICE = 'cuda:0' 
TARGET = "cuda -arch=sm_89"

def run_power_prediction():
    # 1. 파일 경로 설정
    file_path = os.path.expanduser('~/tvm/tutorials/eyas_gpu4090_dataset_densenet169.csv')
    
    if not os.path.exists(file_path):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path}")
        return

    print(f"[INFO] 데이터 로드 중: {file_path}")
    df = pd.read_csv(file_path)

    # 2. 특징(X)과 예측 목표(y) 분리
    drop_cols = ['i', 'model', 'workload_hash', 'trace_hash', 'avg_power_w', 'lat_mean_ms']
    X = df.drop(columns=drop_cols)
    y = df['avg_power_w']

    # 3. 데이터 분할 (8:1:1) - 단순 랜덤 분할 (No Binning)
    indices = np.arange(len(df))
    idx_temp, idx_test, X_temp, X_test, y_temp, y_test = train_test_split(
        indices, X, y, test_size=0.1, random_state=42
    )
    
    idx_train, idx_val, X_train, X_val, y_train, y_val = train_test_split(
        idx_temp, X_temp, y_temp, test_size=0.1111, random_state=42
    )

    print(f"[INFO] 데이터 분할 완료 -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 4. XGBoost 모델 정의 (성능 최적화 하이퍼파라미터)
    model = xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=12,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        colsample_bylevel=0.8,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=1.5,
        tree_method='hist',
        device=DEVICE,
        early_stopping_rounds=100,
        eval_metric='mape',
        random_state=42
    )

    # 5. 학습 시작
    print(f"[INFO] 최적화된 하이퍼파라미터로 학습 시작 (Target: {TARGET})")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # 6. 성능 평가 및 정확도 환산
    best_val_mape = model.best_score * 100
    y_pred = model.predict(X_test)
    test_mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    test_accuracy = 100 - test_mape
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*55)
    print(f"{'성능 지표 리포트 (No Binning)':^55}")
    print("-" * 55)
    print(f"Best Validation MAPE : {best_val_mape:.4f}%")
    print(f"Final Test MAPE      : {test_mape:.4f}%")
    print(f"Final Test Accuracy  : {test_accuracy:.4f}%")
    print(f"R-squared Score      : {r2:.4f}")
    print("=" * 55)

    # 7. 피처 중요도 분석 (Top 10)
    print("\n[INFO] 상위 10개 중요 피처 (전력 예측 기여도):")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False).head(10))

    # 8. 심층 분석 (Worst 10) - 주파수 포함
    abs_error = np.abs(y_test - y_pred)
    worst_idx = np.argsort(abs_error)[-10:][::-1]
    worst_orig_idx = idx_test[worst_idx]

    print("\n[DIAGNOSTICS] 가장 오차가 큰 Top 10 데이터 분석:")
    print("-" * 105)
    print(f"{'순위':<4} | {'주파수(MHz)':<10} | {'실제전력(W)':<10} | {'예측전력(W)':<10} | {'오차율(%)':<8} | {'Workload Hash':<20}")
    print("-" * 105)
    
    for i, (orig_idx, pred) in enumerate(zip(worst_orig_idx, y_pred[worst_idx])):
        row = df.iloc[orig_idx]
        actual = row['avg_power_w']
        freq = row['freq_mhz']
        pct_err = (abs(actual - pred) / actual) * 100
        
        print(f"{i+1:<4} | {freq:<12.1f} | {actual:<12.1f} | {pred:<12.1f} | {pct_err:<10.2f} | {row['workload_hash']}")
    print("-" * 105)

    # 9. 모델 저장
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_save_path = os.path.join('models', 'power_model_sm_89_tuned_no_binning.json')
    model.save_model(model_save_path)
    print(f"\n[INFO] 최적화 모델 저장 완료: {model_save_path}")

if __name__ == "__main__":
    run_power_prediction()