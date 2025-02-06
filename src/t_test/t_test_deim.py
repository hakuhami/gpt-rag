import json
import os
import numpy as np
from scipy import stats

# グローバル変数の定義(これらを変化して検定対象データを変える)
# baseline_dir = "data/deim_output/result/3_chikuji/1_baseline"
# rag_dir = "data/deim_output/result/3_chikuji/miss_explanained_en"
# file_names = [f"{i}_set_evaluation_results.json" for i in range(1, 6)]

### イメージベース評価用
baseline_dir = "image_deim/result/change0_114/1_baseline"
rag_dir = "image_deim/result/change0_114/3_rag"
file_names = [f"{i}_set_evaluation_results.json" for i in range(1, 6)]

def read_f_scores(directory):
   f_scores = []
   for file_name in file_names:
       file_path = directory + "/" + file_name
       with open(file_path, 'r') as f:
           data = json.load(f)
           # 検定対象ラベルは以下を変化し指定する
           f_score = round(data["evidence_quality"]["f"], 3)
           f_scores.append(f_score)
           print(f"File: {file_name}, F-score: {f_score}")
   return np.array(f_scores)

def main():
   print("\nBaseline F-scores:")
   baseline_scores = read_f_scores(baseline_dir)
   
   print("\nRAG F-scores:")
   rag_scores = read_f_scores(rag_dir)
   
   print("\nBasic Statistics:")
   print(f"Baseline - Mean: {np.mean(baseline_scores):.4f}, Std: {np.std(baseline_scores, ddof=1):.4f}")
   print(f"RAG - Mean: {np.mean(rag_scores):.4f}, Std: {np.std(rag_scores, ddof=1):.4f}")
   
   t_stat, p_value = stats.ttest_rel(baseline_scores, rag_scores)
   
   print("\nt-test Results:")
   print(f"t-statistic: {t_stat:.4f}")
   print(f"p-value: {p_value:.4f}")
   print(f"Significant difference (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")

if __name__ == "__main__":
   main()