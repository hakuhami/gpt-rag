import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import json

def read_excel_data(file_path, sheet_name, usecols):
    return pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols, header=None, skiprows=1, nrows=200)

def preprocess_data(df):
    # 列名を設定
    df.columns = ['promise_status_1', 'promise_status_2', 'verification_timeline_1', 'verification_timeline_2',
                  'evidence_status_1', 'evidence_status_2', 'evidence_quality_1', 'evidence_quality_2']
    
    # 空文字列を NaN に変換
    df = df.replace('', np.nan)
    
    # 数値以外の値を NaN に変換
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 小数点を含む値を NaN に変換
    df = df.apply(lambda x: x.where(x.apply(lambda v: v.is_integer() if pd.notnull(v) else True), np.nan))
    
    return df

def calculate_cohen_kappa(data1, data2):
    # 両方のデータが有効な整数である行のみを抽出
    valid_mask = data1.notna() & data2.notna()
    valid_data1 = data1[valid_mask].astype(int)
    valid_data2 = data2[valid_mask].astype(int)
    
    if len(valid_data1) == 0:
        print(f"警告: 有効なデータがありません。")
        return np.nan
    
    return cohen_kappa_score(valid_data1, valid_data2)

# メイン処理
file_path = "./data/Cohen_Japanese/Cohen_2.xlsx"
sheet_name = "Oshima"
usecols = "E,F,H,I,K,L,N,O"

# データ読み込みと前処理
df = read_excel_data(file_path, sheet_name, usecols)
df = preprocess_data(df)

# ラベルとそれに対応する列名を定義
labels = {
    "promise_status": ('promise_status_1', 'promise_status_2'),
    "verification_timeline": ('verification_timeline_1', 'verification_timeline_2'),
    "evidence_status": ('evidence_status_1', 'evidence_status_2'),
    "evidence_quality": ('evidence_quality_1', 'evidence_quality_2')
}

results = {}

for label, (col1, col2) in labels.items():
    kappa = calculate_cohen_kappa(df[col1], df[col2])
    results[label] = kappa

# 結果をJSONファイルに出力
output_file = "./data/Cohen_Japanese/Cohen_2_O.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Cohenのカッパ係数の計算が完了し、結果が{output_file}に保存されました。")
print("計算されたCohenのカッパ係数:")
for label, kappa in results.items():
    print(f"{label}: {kappa}")

# データの検証
print("\nデータサンプル:")
print(df.head())

print("\n各列のデータ型:")
print(df.dtypes)

print("\n各列の一意な値:")
for col in df.columns:
    unique_values = sorted(df[col].dropna().unique())
    print(f"{col}: {unique_values}")
    print(f"  空値の数: {df[col].isna().sum()}")

print("\n各ラベルの有効なデータ数:")
for label, (col1, col2) in labels.items():
    valid_count = (df[col1].notna() & df[col2].notna()).sum()
    total_count = len(df)
    print(f"{label}: {valid_count}/{total_count} ({valid_count/total_count:.2%})")
