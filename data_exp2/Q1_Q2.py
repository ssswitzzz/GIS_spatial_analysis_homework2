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


fa = FactorAnalyzer(n_factors=4, rotation='varimax', method='principal')
fa.fit(data_for_pca)
# 1. 获取“提取”列的数据 (即公因子方差)
# communalities = fa.get_communalities()
#
# # 2. 构建表格
# # 变量名列表 (假设 data_for_pca 是你的数据框)
# variable_names = data_for_pca.columns
#
# # 创建 DataFrame
# df_communalities = pd.DataFrame({
#     '变量': variable_names,
#     '初始': 1.0,  # PCA 方法下，初始公因子方差默认为 1
#     '提取': communalities
# })
#
# # 3. 打印结果 (保留3位小数，模拟 SPSS 格式)
# print("-" * 30)
# print("公因子方差 (Communalities)")
# print("-" * 30)
# print(df_communalities.set_index('变量').round(3))
# df_communalities.round(3).to_excel('公因子方差表.xlsx')
#
# # --- 3. 解释的总方差 (作业 Page 10) ---
# ev, v, cv = fa.get_factor_variance()
# var_table = pd.DataFrame(data={'特征值': ev, '方差贡献率': v, '累积方差贡献率': cv},
#                          index=['F1', 'F2', 'F3', 'F4'])
# print("\n解释的总方差:")
# print(var_table)
#
# # --- 4. 旋转后的成分矩阵 (作业 Page 11) ---
# rotated_matrix = pd.DataFrame(fa.loadings_, index=data_for_pca.columns, columns=['F1', 'F2', 'F3', 'F4'])
# # 筛选大于 0.5 的载荷以便查看（选做）
# print("\n旋转后的成分矩阵 (Loadings):")
# print(rotated_matrix.round(3))
# rotated_matrix.to_excel('旋转成分矩阵.xlsx')

# -----------------------------------------------------------
# 第一步：准备工作 (确保符号正确，跟PPT一致)
# -----------------------------------------------------------
# 1. 获取旋转后的成分矩阵 (Loadings)
loadings_df = pd.DataFrame(fa.loadings_, index=data_for_pca.columns, columns=['F1', 'F2', 'F3', 'F4'])

# 2. 自动修正符号 (这一步很重要！确保主要指标的载荷是正的)
# 检查每一列，如果绝对值最大的载荷是负数，就整列乘以 -1

# 3. 计算成分得分系数矩阵 (Score Coefficient Matrix)
# 公式: R逆 * 载荷矩阵
corr_matrix = data_for_pca.corr()
score_coef_df = pd.DataFrame(
    np.linalg.inv(corr_matrix).dot(loadings_df),
    index=data_for_pca.columns,
    columns=['F1', 'F2', 'F3', 'F4']
)

print("修正后的旋转成分矩阵 (Loadings):")
print(loadings_df.round(3))
print("\n成分得分系数矩阵 (Score Coefficients):")
print(score_coef_df.round(3))

# -----------------------------------------------------------
# 第二步：按作业要求筛选指标并计算得分 (核心修改)
# -----------------------------------------------------------
# 创建一个空的 DataFrame 存得分
scores = pd.DataFrame(index=data_for_pca.index)
for i in range(4):
    factor_name = f'F{i + 1}'
    score_col_name = f'C{i + 1}'

    # 1. 筛选：找出该因子下，旋转载荷绝对值 > 0.7 的变量
    # 注意：这里是用【旋转成分矩阵】来判断“谁归谁管”
    relevant_vars = loadings_df[loadings_df[factor_name].abs() > 0.7].index.tolist()

    print(f"  - {score_col_name} 由以下指标构成: {relevant_vars}")

    subset_data = data_for_pca[relevant_vars]
    # 获取这些变量在该因子下的得分系数 (不是载荷，是得分系数！)
    subset_coefs = score_coef_df.loc[relevant_vars, factor_name]

    # 执行加权求和
    scores[score_col_name] = subset_data.dot(subset_coefs)

# -----------------------------------------------------------
# 第三步：计算综合得分 CI
# -----------------------------------------------------------
# 获取特征值 (解释方差)
ev, _ = fa.get_eigenvalues()
ev = ev[:4]  # 只取前4个
print("\n特征值 (用于计算CI的权重):", ev)

# CI = C1*特征值1 + C2*特征值2 ...
scores['CI'] = 0
for i in range(4):
    scores['CI'] += scores[f'C{i + 1}'] * ev[i]

# 合并结果
final_result = pd.concat([df[['PAC', 'NAME']], scores], axis=1)
print("\n最终结果预览:")
print(final_result.head())

final_result.to_excel('4_PCA计算结果_按作业筛选版.xlsx', index=False)