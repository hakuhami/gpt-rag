import numpy as np
from scipy import stats

# 各手法のF値を設定
baseline = np.array([0.825, 0.428, 0.649, 0.411, 0.719, 0.394])  # ベースライン手法のF値
teian = np.array([0.825, 0.558, 0.876, 0.455, 0.747, 0.469])    # 提案手法のF値

# 基本統計量の計算と表示
print("基本統計量:")
print(f"ベースライン手法 - 平均: {np.mean(baseline):.4f}, 標準偏差: {np.std(baseline, ddof=1):.4f}")
print(f"提案手法 - 平均: {np.mean(teian):.4f}, 標準偏差: {np.std(teian, ddof=1):.4f}")

# 両側t検定の実行（有意水準5%）
# alternative='two-sided'で両側検定を明示的に指定
t_stat, p_value = stats.ttest_ind(baseline, teian, alternative='two-sided')

# 効果量（Cohen's d）の計算
n1, n2 = len(baseline), len(teian)
pooled_std = np.sqrt(((n1 - 1) * np.std(baseline, ddof=1)**2 + 
                     (n2 - 1) * np.std(teian, ddof=1)**2) / (n1 + n2 - 2))
cohens_d = (np.mean(teian) - np.mean(baseline)) / pooled_std

# 結果の表示
print("\n両側t検定の結果（有意水準5%）:")
print(f"t値: {t_stat:.4f}")
print(f"p値: {p_value:.4f}")
print(f"Cohen's d: {cohens_d:.4f}")

# 有意差の判定（有意水準5%）
alpha = 0.05  # 有意水準を明示的に設定
if p_value < alpha:
    print(f"\n※ 有意水準{alpha*100}%で統計的有意差が認められました")
    print(f"   (p値 = {p_value:.4f} < {alpha})")
else:
    print(f"\n※ 有意水準{alpha*100}%で統計的有意差は認められませんでした")
    print(f"   (p値 = {p_value:.4f} ≧ {alpha})")

# 自由度の表示を追加
df = n1 + n2 - 2
print(f"\n自由度: {df}")

# 検定の詳細情報
print("\n検定の詳細:")
print("- 検定の種類: 対応のない両側t検定")
print(f"- 有意水準: {alpha}")
print("- 帰無仮説: 2つの手法の母平均に差がない")
print("- 対立仮説: 2つの手法の母平均に差がある")