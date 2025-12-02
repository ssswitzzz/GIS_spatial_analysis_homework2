import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np


plt.rcParams['font.sans-serif'] = ['STZhongsong']

file_path = '4_PCA计算结果_最终最终版.xlsx'
df = pd.read_excel(file_path)
cluster_data = df[['C1', 'C2', 'C3', 'C4']].copy()
cluster_data.index = df['NAME']

# --- 2. 进行系统聚类 (Hierarchical Clustering) ---
Z = linkage(cluster_data, method='ward', metric='euclidean')
# --- 3. 绘制树状图 (对应作业 Page 18) ---
plt.figure(figsize=(12, 6))
plt.title('系统聚类树状图 (Dendrogram)')
plt.xlabel('城市名称')
plt.ylabel('距离 (Distance)')
dendrogram(Z, labels=cluster_data.index, leaf_rotation=90, leaf_font_size=8)
plt.tight_layout()
plt.savefig('5_树状图.png', dpi=300) # 保存图片放报告里
plt.show()

# --- 4. 确定聚类数量 (对应作业 Page 19 的折线图) ---
# SPSS 的“系数”对应 linkage 矩阵的第三列 (距离)。
# 我们画出最后 15 次合并的距离变化，寻找“肘部” (Elbow)
last = Z[-10:, 2]
last_rev = last[::-1]
idxs = range(1, len(last) + 1)

plt.figure(figsize=(8, 4))
plt.plot(idxs, last_rev, marker='o', color='b')
plt.title('聚类系数变化图 (Elbow Method)')
plt.xlabel('聚类数量 (K)')
plt.ylabel('合并距离 (Coefficient)')
plt.grid(True)
plt.xticks(idxs)
plt.savefig('5_聚类系数折线图.png', dpi=300) # 保存图片放报告里
plt.show()
print("提示：观察折线图，如果 K=4 处有明显的转折（斜率变缓），则说明分4类是合适的。")

# --- 5. 保存聚类结果 (K=4) ---
# 作业 Page 20 明确要求聚类数为 4
k = 4
df['Cluster_Label'] = fcluster(Z, k, criterion='maxclust')

# --- 6. 结果分析：计算每一类的均值 (对应作业 Page 21) ---
# 我们需要算出每一类在 C1, C2, C3, C4 以及 CI 上的平均分
cluster_profile = df.groupby('Cluster_Label')[['C1', 'C2', 'C3', 'C4', 'CI']].mean()

# 按 CI (综合指数) 对聚类进行排序，方便命名 (最低 -> 最高)
cluster_profile_sorted = cluster_profile.sort_values(by='CI')
print("\n--- 各聚类类别的平均特征 (按 CI 排序) ---")
print(cluster_profile_sorted.round(3))

# --- 7. 自动打标签 (可选) ---
# 根据 CI 均值大小，给它们起名：最低、较低、较高、最高
rank_map = {idx: label for idx, label in zip(cluster_profile_sorted.index, ['最低', '较低', '较高', '最高'])}
df['弱势性等级'] = df['Cluster_Label'].map(rank_map)

# --- 8. 导出最终数据供 ArcGIS 使用 ---
output_file = '5_聚类分析结果_For_ArcGIS.xlsx'
df.to_excel(output_file, index=False)
print(f"\n最终文件已保存为: {output_file}")
print("请在 ArcGIS Pro 中使用 'Add Join' 将此表连接到你的 shapefile 上，字段选择 'PAC'。")