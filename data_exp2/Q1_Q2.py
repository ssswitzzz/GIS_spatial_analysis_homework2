from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import numpy as np
import pandas as pd

df = pd.read_excel('社会经济指标.xlsx')
desc_stats = df.drop(columns=['PAC','离婚率','无业率','无工作能力人口比例']).describe().T[['min', 'max', 'mean', 'std']]
desc_stats.columns = ['最小值', '最大值', '均值', '标准差']
desc_stats.to_excel('1_描述性统计结果.xlsx')

negative_cols = ['年平均工资', '白领比例', '大专及以上学历', '平均受教育年限']
positive_cols = [col for col in df.columns if col not in negative_cols + ['PAC', 'NAME']]
df_norm = df.copy()

# 标准化
for col in positive_cols:
    df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
for col in negative_cols:
    df_norm[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min())

df_norm.to_excel('2_标准化结果.xlsx', index=False)

data_for_pca = df_norm.drop(columns=['PAC', 'NAME','离婚率','无业率','无工作能力人口比例'])

# --- 1. KMO 和 Bartlett 球形度检验 (作业 Page 10) ---
chi_square_value, p_value = calculate_bartlett_sphericity(data_for_pca)
kmo_all, kmo_model = calculate_kmo(data_for_pca)
p = data_for_pca.shape[1]  # 变量个数，应该是 13
df_degrees = (p * (p - 1)) / 2
print(f'Bartlett 球形度检验结果:\n卡方值: {chi_square_value:.3f}, 自由度: {df_degrees}, p 值: {p_value:.3f}')
print(f'KMO 检验结果:\n整体 KMO 值: {kmo_model:.3f}')
# 截图这些输出贴到作业里的 KMO 表格位置

# --- 2. 主成分分析 (复刻 SPSS 设置) ---
# 作业提示提取 4 个主成分，并使用最大方差旋转 (Varimax)
# method='principal' 对应 SPSS 的“主成分法”
fa = FactorAnalyzer(n_factors=4, rotation='varimax', method='principal')
fa.fit(data_for_pca)
# 1. 获取“提取”列的数据 (即公因子方差)
communalities = fa.get_communalities()

# 2. 构建表格
# 变量名列表 (假设 data_for_pca 是你的数据框)
variable_names = data_for_pca.columns

# 创建 DataFrame
df_communalities = pd.DataFrame({
    '变量': variable_names,
    '初始': 1.0,  # PCA 方法下，初始公因子方差默认为 1
    '提取': communalities
})

# 3. 打印结果 (保留3位小数，模拟 SPSS 格式)
print("-" * 30)
print("公因子方差 (Communalities)")
print("-" * 30)
print(df_communalities.set_index('变量').round(3))
df_communalities.round(3).to_excel('公因子方差表.xlsx')

# --- 3. 解释的总方差 (作业 Page 10) ---
ev, v, cv = fa.get_factor_variance()
var_table = pd.DataFrame(data={'特征值': ev, '方差贡献率': v, '累积方差贡献率': cv},
                         index=['F1', 'F2', 'F3', 'F4'])
print("\n解释的总方差:")
print(var_table)

# --- 4. 旋转后的成分矩阵 (作业 Page 11) ---
rotated_matrix = pd.DataFrame(fa.loadings_, index=data_for_pca.columns, columns=['F1', 'F2', 'F3', 'F4'])
# 筛选大于 0.5 的载荷以便查看（选做）
print("\n旋转后的成分矩阵 (Loadings):")
print(rotated_matrix.round(3))
rotated_matrix.to_excel('旋转成分矩阵.xlsx')

# --- 5. 计算各主成分得分 & 综合得分 CI (作业 Page 12) ---
# factor_analyzer 的 transform 方法可以直接计算因子得分 (Regression method by default)
# 注意：SPSS 的因子得分系数矩阵和 Python 计算逻辑略有不同，但 transform 的结果是直接可用的得分。
factor_scores = fa.transform(data_for_pca)
df_scores = pd.DataFrame(factor_scores, columns=['C1', 'C2', 'C3', 'C4'])

# 计算综合得分 CI
# CI = (C1*方差1 + C2*方差2 + C3*方差3 + C4*方差4) / 累积方差
weights = var_table.loc['F1':'F4', '方差贡献率'].values
df_scores['CI'] = np.dot(df_scores[['C1', 'C2', 'C3', 'C4']], weights) / weights.sum()

# 合并回原始信息
final_result = pd.concat([df[['PAC', 'NAME']], df_scores], axis=1)

# --- 6. 导出结果 ---
final_result.to_excel('4_PCA计算结果_最终.xlsx', index=False)