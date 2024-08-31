import dask.dataframe as dd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import shap


df = dd.read_csv('E://table//JIANMO//附件data//train.csv',encoding="gbk")

columns_to_analyze = df.columns.difference(['id', '洪水概率'])

sample_fraction = 0.01
sample_df = df.sample(frac=sample_fraction).compute()

X = sample_df[columns_to_analyze].values
y = sample_df['洪水概率'].values

rf = RandomForestRegressor(n_jobs=-1, max_depth=5)

boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

boruta_selector.fit(X, y)

important_features = columns_to_analyze[boruta_selector.support_].to_list()
print("Important Features：", important_features)

rf.fit(X[:, boruta_selector.support_], y)

# SHAP
explainer = shap.Explainer(rf)
shap_values = explainer(X[:, boruta_selector.support_])

shap.summary_plot(shap_values, feature_names=important_features)


#
import dask.dataframe as dd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt

df = dd.read_csv('E://table//JIANMO//附件data//train_run1.csv', encoding="gbk")
train_df = df.drop(['id'], axis=1).compute()  # Convert Dask DataFrame to Pandas DataFrame

correlation_matrix = train_df.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Matrix Heatmap')
plt.savefig('E://table//JIANMO//correlation_matrix_heatmap.png')  # Save the heatmap as an image file
plt.close()  # Close the figure to free up memory


#第二题
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv('E://table//JIANMO//附件data//train.csv', encoding="gbk")

columns_to_analyze = train_df.columns.difference(['id', 'probability'])

kmeans = KMeans(n_clusters=3, random_state=42)
train_df['risk_category'] = kmeans.fit_predict(train_df[['probability']])

train_df.to_csv("E://table//JIANMO//附件data//risk_category.csv")

X = train_df.drop(['probability', 'risk_category'], axis=1)
y = train_df['probability']

rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)

feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
important_features = feature_importances.sort_values(ascending=False).head(10)
print("Important Features：\n", important_features)

#lasso计算权重，并建立预警评价模型
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

train_df = pd.read_csv('E://table//JIANMO//附件data//risk_category.csv', encoding="gbk")

important_features = [
    "name12", "name20", "name13", "name9", "name8",
    "name5", "name14", "name4", "name18", "name10"
]

for category in range(3):
    category_df = train_df[train_df['risk_category'] == category]
    X_category = category_df[important_features]
    y_category = category_df['probability']

    scaler = StandardScaler()
    X_category_scaled = scaler.fit_transform(X_category)

    lasso = Lasso(alpha=0.1, random_state=2024)
    lasso.fit(X_category_scaled, y_category)

    sensitivity_analysis = pd.Series(lasso.coef_, index=important_features)
    print(f"灵敏度分析{category}：\n", sensitivity_analysis)

    feature_weights = pd.Series(lasso.coef_, index=important_features)
    print(f"风险类别 {category} 的特征权重：\n", feature_weights)

    explainer = shap.Explainer(lasso, X_category_scaled)
    shap_values = explainer(X_category_scaled)
    shap.summary_plot(shap_values, feature_names=important_features)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_names=important_features, plot_size=(10, 6))
    plt.tight_layout()
    plt.savefig(f'E://table//JIANMO//风险类别_{category}_SHAP.png',dpi=300)

   