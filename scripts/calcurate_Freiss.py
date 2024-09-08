# 1つのファイルにまとめられていない

import pandas as pd
import numpy as np
import json

def fleiss_kappa(ratings):
    n_raters = ratings.shape[1]
    n_items = ratings.shape[0]
    
    # すべての値を文字列に変換（NaNは'nan'として扱う）
    str_ratings = ratings.astype(str)
    
    # カテゴリの数を動的に決定
    categories = np.unique(str_ratings)
    n_categories = len(categories)
    
    # カテゴリの再マッピング
    category_mapping = {cat: i for i, cat in enumerate(categories)}
    
    def map_category(x):
        return category_mapping.get(x, -1)
    
    mapped_ratings = np.vectorize(map_category)(str_ratings)
    
    # カテゴリごとの評価者数をカウント
    category_counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=n_categories), 
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
    # データをそのまま使用
    ratings = sheet.values
    return fleiss_kappa(ratings)

# Excelファイルを読み込む
file_path = "./data/Freiss_Japanese/Freiss_1_final.xlsx"
sheet_name = "sample200"

# 指定された列（H, I, J）のデータを読み込む
df = pd.read_excel(file_path, sheet_name=sheet_name, usecols="AL:AN", header=None)

# Fleissのカッパ係数を計算
kappa = process_sheet(df)

# 結果を辞書形式で保存
results = {sheet_name: kappa}

# 結果をJSONファイルに出力
with open("./data/Freiss_Japanese/Freiss_1_final_eq.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"Fleissのカッパ係数の計算が完了し、結果が./data/Freiss_1.jsonに保存されました。")
print(f"計算されたFleissのカッパ係数: {kappa}")