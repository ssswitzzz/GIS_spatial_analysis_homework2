# import pandas as pd
#
# # 1. 读取数据
# file_path = '聚类结果.xlsx'  # 请根据实际文件名修改
# df = pd.read_excel(file_path)
#
# # 2. 计算各类的均值
# cluster_stats = df.groupby('CLU4_1')[['C1', 'C2', 'C3', 'C4', 'CI']].mean()
#
# # 3. 生成特征描述和括号格式
# # 逻辑：数值越大 -> 排名越靠前 (Rank 1) -> "最高"
# ranks = cluster_stats[['C1', 'C2', 'C3', 'C4']].rank(ascending=False)
# rank_map = {1: "最高", 2: "较高", 3: "较低", 4: "最低"}
#
# # 存储格式化后的列
# formatted_cols = {}
# descriptions = []
#
# # 遍历每一行
# for idx, row in cluster_stats.iterrows():
#     desc_list = []
#     for col in ['C1', 'C2', 'C3', 'C4']:
#         # 获取均值和排名
#         mean_val = row[col]
#         rank_val = ranks.loc[idx, col]
#         label = rank_map[int(rank_val)]
#
#         # 生成描述列表
#         desc_list.append(label)
#
#         # 生成题目要求的格式：数值 (评价)
#         # 如果是第一次遍历该列，初始化list
#         if col not in formatted_cols:
#             formatted_cols[col] = []
#         formatted_cols[col].append(f"{mean_val:.4f} ({label})")
#
#     descriptions.append("-".join(desc_list))
#
# # 4. 构建最终表格
# # 使用格式化后的数据替换原始数据，但保留CI用于排序
# final_df = pd.DataFrame(formatted_cols, index=cluster_stats.index)
# final_df['CI'] = cluster_stats['CI']
# final_df['特征描述'] = descriptions
# final_df['包含城市'] = df.groupby('CLU4_1')['NAME'].apply(lambda x: ' '.join(x))
#
# # 5. 按照 CI 排序 (你的思路，很棒)
# final_df = final_df.sort_values(by='CI', ascending=False)
#
# # 6. 调整列顺序，符合阅读习惯
# cols = ['特征描述', 'C1', 'C2', 'C3', 'C4', 'CI', '包含城市']
# final_df = final_df[cols]
#
# print(">>> 最终修正结果 (按综合弱势度CI降序):")
# print(final_df.to_string())
#
# # 保存
# final_df.to_excel("最终聚类分析表.xlsx")

import pandas as pd


def add_cluster_description_column():
    # 1. 读取原始数据
    file_path = '聚类结果.xlsx'
    df = pd.read_excel(file_path)

    # 2. 计算各类的均值并生成映射关系
    # 按照 CLU4_1 分组求均值
    cluster_means = df.groupby('CLU4_1')[['C1', 'C2', 'C3', 'C4']].mean()

    # 计算排名: 数值越大，排名越前 (1为最高)
    ranks = cluster_means.rank(ascending=False)

    # 定义排名到文字的映射
    rank_to_text = {1.0: "最高", 2.0: "较高", 3.0: "较低", 4.0: "最低"}

    # 生成 {类别ID : 特征描述字符串} 的字典
    id_to_desc_map = {}
    for cluster_id, row in ranks.iterrows():
        desc_parts = [rank_to_text[row[col]] for col in ['C1', 'C2', 'C3', 'C4']]
        # 用短横线连接，如 "较高-较高-最高-较低"
        id_to_desc_map[cluster_id] = "-".join(desc_parts)

    print("生成的类别映射关系：")
    for k, v in id_to_desc_map.items():
        print(f"类别 {k} -> {v}")

    # 3. 在原数据表中新建一列 '聚类结果'
    # 使用 map 函数根据 CLU4_1 的值自动填入对应的字符串
    df['聚类结果'] = df['CLU4_1'].map(id_to_desc_map)

    # 4. 保存结果
    output_file = '聚类结果_最终版.xlsx'
    df.to_excel(output_file, index=False)
    print(f"\n处理完成！文件已保存为: {output_file}")

    # 打印前几行预览
    print("\n数据预览:")
    print(df[['PAC', 'NAME', 'CLU4_1', '聚类结果']].head())


if __name__ == "__main__":
    add_cluster_description_column()