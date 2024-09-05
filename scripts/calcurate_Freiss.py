import pandas as pd
import numpy as np
import json
from scipy import stats

def fleiss_kappa(ratings):
    n_raters = ratings.shape[1]
    n_items = ratings.shape[0]
    
    # カテゴリの数を動的に決定（0以上の整数値のみを考慮）
    categories = np.unique(ratings[ratings >= 0])
    n_categories = len(categories)
    
    # カテゴリの再マッピング
    category_mapping = {cat: i for i, cat in enumerate(categories)}
    mapped_ratings = np.vectorize(lambda x: category_mapping.get(x, -1))(ratings)
    
    # カテゴリごとの評価者数をカウント（-1はカウントしない）
    category_counts = np.apply_along_axis(
        lambda x: np.bincount(x[x >= 0], minlength=n_categories), 
        axis=1, 
        arr=mapped_ratings
    )
    
    # 項目ごとの一致度を計算
    p_i = np.sum(category_counts * (category_counts - 1), axis=1) / (n_raters * (n_raters - 1))
    P_bar = np.mean(p_i)

    # カテゴリごとの周辺確率を計算
    p_j = np.sum(category_counts, axis=0) / (n_items * n_raters)
    P_e = np.sum(p_j ** 2)

    # Fleissのカッパ係数を計算
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa

def process_sheet(sheet):
    # NaNを含む行を削除
    sheet_clean = sheet.dropna()
    
    # データを整数に変換（小数点以下は切り捨て）
    ratings = sheet_clean.astype(float).astype(int).values
    
    return fleiss_kappa(ratings)

# Excelファイルを読み込む
file_path = "./data/Freiss_Japanese/Freiss_1.xlsx"
sheets = ["promise_status", "verification_timeline", "evidence_status", "evidence_quality"]

results = {}

for sheet_name in sheets:
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols="A:C", header=None)
    kappa = process_sheet(df)
    results[sheet_name] = kappa

# 結果をJSONファイルに出力
with open("./data/Freiss_Japanese/Freiss_1.json", "w") as f:
    json.dump(results, f, indent=4)

print("Fleissのカッパ係数の計算が完了し、結果が./data/Freiss_1.jsonに保存されました。")