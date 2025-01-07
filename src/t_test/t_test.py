import numpy as np
from scipy import stats

## 1段階手法について
baseline = np.array([0.825, 0.428, 0.649, 0.411, 0.719, 0.394])  # ベースライン手法のF値
# 1段階手法のRAG有り
teian = np.array([0.825, 0.558, 0.876, 0.455, 0.747, 0.469])
# # 1段階手法のRAGと解説文有り
# teian = np.array([0.862, 0.507, 0.852, 0.427, 0.769, 0.482])

# ## 3段階手法について
# baseline = np.array([0.825, 0.456, 0.715, 0.409, 0.709, 0.496])  # ベースライン手法のF値
# # 3段階手法のRAG有り
# teian = np.array([0.844, 0.561, 0.860, 0.509, 0.735, 0.567])
# # 3段階手法のRAGと解説文有り
# teian = np.array([0.878, 0.561, 0.914, 0.465, 0.726, 0.599])

# ## 4段階手法について
# baseline = np.array([0.830, 0.225, 0.628, 0.370, 0.702, 0.483])  # ベースライン手法のF値
# # 4段階手法のRAG有り
# teian = np.array([0.853, 0.466, 0.815, 0.411, 0.738, 0.613])
# # 4段階手法のRAGと解説文有り
# teian = np.array([0.869, 0.442, 0.843, 0.324, 0.749, 0.618])


# 1対の対応のある両側t検定の実行
t_stat, p_value = stats.ttest_rel(baseline, teian)

# 効果量（Pearson's r）の計算
df = len(baseline) - 1
r = np.abs(t_stat) / np.sqrt(t_stat**2 + df)

# 記述統計量の計算
baseline_mean = np.mean(baseline)
teian_mean = np.mean(teian)
baseline_std = np.std(baseline, ddof=1)
teian_std = np.std(teian, ddof=1)

# 効果量の解釈
def interpret_effect_size(r):
    r = abs(r)
    if r < 0.1:
        return "無視できる効果量"
    elif r < 0.3:
        return "小さい効果量"
    elif r < 0.5:
        return "中程度の効果量"
    else:
        return "大きい効果量"

# 結果の出力
print("==== 統計分析結果 ====")
print(f"t統計量: {t_stat:.4f}")
print(f"p値: {p_value:.4f}")
print(f"効果量 (Pearson's r): {r:.4f}")
print(f"有意差: {'あり' if p_value < 0.05 else 'なし'}")

print("\n==== 記述統計量 ====")
print(f"ベースライン手法の平均F値: {baseline_mean:.4f}")
print(f"提案手法の平均F値: {teian_mean:.4f}")
print(f"ベースライン手法の標準偏差: {baseline_std:.4f}")
print(f"提案手法の標準偏差: {teian_std:.4f}")

print(f"\n効果量の解釈: {interpret_effect_size(r)}")