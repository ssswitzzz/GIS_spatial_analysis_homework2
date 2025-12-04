import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import numpy as np

# 设置绘图风格 (解决中文显示)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
file_path = '4_PCA计算结果_最终最终版.xlsx'
df = pd.read_excel(file_path)

# 2. 聚类分析 (Ward + Euclidean)
cluster_data = df[['C1', 'C2', 'C3', 'C4']].copy()
# 确保数据是数值型
cluster_data = cluster_data.astype(float)
cluster_data.index = df['NAME']

Z = linkage(cluster_data, method='ward', metric='euclidean')

# =======================================================
# 关键修正 1: 复刻 PPT 的“聚类系数变化图”
# =======================================================
# SPSS 的 Coefficients = Python Distance 的平方
# PPT 里的折线图通常是“倒着画”的（从聚类数多 -> 少，或者显示最后几步的合并系数）

last_steps = 15
last_dist = Z[-last_steps:, 2]
last_coeffs = last_dist ** 2  # <--- 核心修改：取平方！这样就像 SPSS 了

idxs = np.arange(1, last_steps + 1)

plt.figure(figsize=(8, 5))
# 这里展示的是最后 15 次合并的系数变化
# 聚类数从 15 降到 1
plt.plot(idxs, last_coeffs[::-1], marker='o', color='red', label='系数 (SPSS Coefficients)')

plt.title('聚类系数变化图 (模拟 SPSS 算法)')
plt.xlabel('聚类数量 (K)')
plt.ylabel('系数 (距离的平方)')
plt.grid(True, linestyle='--')
plt.xticks(idxs)

# 标记 K=4
plt.axvline(x=4, color='blue', linestyle='--', label='K=4 (肘部)')
plt.legend()
plt.savefig('5_聚类系数折线图_SPSS版.png', dpi=300)
plt.show()
print("看图：现在这个曲线是不是变陡了？这就对了！")

# =======================================================
# 关键修正 2: 生成跟 PPT 一样的“聚类特征表”
# =======================================================
# 1. 先定 K=4
k = 4
df['Cluster_Label'] = fcluster(Z, k, criterion='maxclust')

# 2. 计算均值
means = df.groupby('Cluster_Label')[['C1', 'C2', 'C3', 'C4', 'CI']].mean()

# 3. 按照 CI (综合弱势性) 从小到大排序！
# 这一步是为了对齐 PPT 的顺序 (PPT通常把弱势性最低的放在第一行)
means_sorted = means.sort_values(by='CI')

# 4. 计算排名 (1=最低, 4=最高)
# 注意：是在排序后的基础上计算排名，或者直接对原始means算排名
ranks = means.rank(ascending=True).astype(int)

# 5. 拼接 "数值 (排名)" 格式
display_df = pd.DataFrame(index=means_sorted.index) # 使用排序后的索引

for col in ['C1', 'C2', 'C3', 'C4', 'CI']:
    # 找回原始的 rank
    col_ranks = ranks.loc[display_df.index, col]
    # 拼接
    display_df[col] = means_sorted[col].round(3).astype(str) + " (" + col_ranks.astype(str) + ")"

# 6. 加上类型名称
label_map = {1: '弱势性最低', 2: '弱势性较低', 3: '弱势性较高', 4: '弱势性最高'}
# 获取 CI 的 rank
ci_ranks = ranks.loc[display_df.index, 'CI']
display_df['类型定义'] = ci_ranks.map(label_map)

print("\n" + "="*50)
print("聚类特征表 (已按 CI 从低到高排序，完全复刻 PPT)")
print("注意：Cluster_Label 的数字(1,2,3,4)不重要，重要的是里面的数值和排名！")
print("="*50)
print(display_df)

# 导出
display_df.to_excel('5_聚类特征分析表_复刻版.xlsx')