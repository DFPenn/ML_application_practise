import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_excel("E:\\table\\task2.xlsx")
X = data.iloc[:, 3:]  # 选择数值特征列

# 标准化数据
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 执行PCA
pca = PCA()
pca.fit(X_std)

# 查看每个主成分解释的方差比例
explained_variance_ratio = pca.explained_variance_ratio_
print(f"每个主成分解释的方差比例:\n{explained_variance_ratio}")

# 确定需要保留的主成分数量,使损失信息不超过20%
cumulative_variance_ratio = explained_variance_ratio.cumsum()
n_components = len(cumulative_variance_ratio[cumulative_variance_ratio < 0.8])
print(f"\n为了使损失信息不超过20%,需要保留{n_components}个主成分。")

# 计算主成分得分
X_pca = pca.transform(X_std)[:, :n_components]
principal_component_scores = pd.DataFrame(X_pca, index=X.index, columns=[f"PC{i+1}" for i in range(n_components)])

# 对主成分得分进行排序和分类
print("\n主成分得分排序:")
print(principal_component_scores.sort_values(by="PC1", ascending=False))


# 对主成分得分进行聚类分析(此处简单示例,使用第一主成分进行分类)
median_pc1 = principal_component_scores["PC1"].median()
cluster1 = principal_component_scores[principal_component_scores["PC1"] >= median_pc1].index
cluster2 = principal_component_scores[principal_component_scores["PC1"] < median_pc1].index

print("\n基于第一主成分的聚类结果:")
print("群集1:", list(cluster1))
print("群集2:", list(cluster2))