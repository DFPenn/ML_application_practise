import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve,auc
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('E://table//JIANMO//附件data//train.csv', encoding="gbk")

features = ["基础设施恶化", "大坝质量", "淤积", "政策因素", "地形排水",
            "河流管理", "滑坡", "季风强度", "无效防灾", "气候变化"]
X = df[features]
y = df['洪水概率']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2024)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.01,reg_alpha=0.1,
                             max_depth=10, booster="gbtree", random_state=2024)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

def plot_sample_scatter(y_true, y_pred, sample_idx, iteration):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3)
    plt.xlabel('Actual Flood Probability')
    plt.ylabel('Predicted Flood Probability')
    plt.title(f'Actual vs Predicted Flood Probability (Sample {iteration + 1})')
    plt.savefig(f'E://table//预测表现_Sample_{iteration + 1}.png')
    plt.close()

# 不放回随机抽取10次，每次抽取100个样本并绘制散点图
np.random.seed(2024)
sample_size = 100

for i in range(10):
    sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
    plot_sample_scatter(y_test.values, y_pred, sample_idx, i)


# 将预测的概率转换为二分类标签
threshold = 0.5
y_pred_class = (y_pred >= threshold).astype(int)
y_test_class = (y_test >= threshold).astype(int)

roc_auc = roc_auc_score(y_test_class, y_pred)
accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)
print(f"ROC-AUC Score: {roc_auc}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

fpr, tpr, _ = roc_curve(y_test_class, y_pred)
roc_auc_value = auc(fpr, tpr)

# plit roc
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("E://table//预测表现ROC.png")


# PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

X_pcaD=pd.DataFrame(X_pca)
X_pcaD.to_csv("E://table//PCA_variable.csv")

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=2024)

xgb_model_pca =  xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.01,reg_alpha=0.1,
                             max_depth=10, booster="gbtree", random_state=2024)

xgb_model_pca.fit(X_train_pca, y_train_pca)

y_pred_pca = xgb_model_pca.predict(X_test_pca)

# MSE
mse_pca = mean_squared_error(y_test_pca, y_pred_pca)
print(f"PCA_Mean Squared Error (MSE): {mse}")

# RMSE
rmse_pca = mean_squared_error(y_test_pca, y_pred_pca, squared=False)
print(f"PCA_Root Mean Squared Error (RMSE): {rmse}")

# MAE
mae_pca = mean_absolute_error(y_test_pca, y_pred_pca)
print(f"PCA_Mean Absolute Error (MAE): {mae}")

#plot scatter
def plot_PCA_sample_scatter(y_true, y_pred, sample_idx, iteration):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3, color="#29B4B6")
    plt.xlabel('Actual Flood Probability')
    plt.ylabel('PCA_Predicted Flood Probability')
    plt.title(f'Actual vs Predicted Flood Probability (Sample {iteration + 1})')
    plt.savefig(f'E://table//PCA_预测表现_Sample_{iteration + 1}.png')
    plt.close()

# 不放回随机抽取10次，每次抽取100个样本并绘制散点图
np.random.seed(2024)
sample_size = 100

for i in range(10):
    sample_idx = np.random.choice(len(y_test_pca), sample_size, replace=False)
    plot_PCA_sample_scatter(y_test_pca.values, y_pred_pca, sample_idx, i)

#第四问

# 读取测试数据
df_test = pd.read_csv('E://table//JIANMO//附件data//test.csv', encoding="gbk")

X_test = df_test[features]

# 使用模型预测洪水发生概率
y_pred_test = xgb_model.predict(X_test)

df_submit = pd.DataFrame()
df_submit['预测洪水概率'] = y_pred_test
df_submit.to_csv('E://table//submit.csv', index=False, encoding='utf-8-sig')

#plot hist
plt.figure(figsize=(12, 12))
plt.hist(y_pred_test, bins=50, alpha=0.7, color='#F2DB96')
plt.xlabel('Predicted Flood Probability')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Flood Probability')
plt.savefig("E://table//test_histogram.png")

# plot line
plt.figure(figsize=(8, 6))
plt.plot(np.sort(y_pred_test), marker='o', linestyle='-', color='#DE9F83')
plt.xlabel('Event Index')
plt.ylabel('Predicted Flood Probability')
plt.title('Line Plot of Predicted Flood Probability')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("E://table//line_histogram.png")

#normally distribution test
import scipy.stats as stats
# Shapiro-Wilk
shapiro_test_stat, shapiro_p_value = stats.shapiro(y_pred_test)
print(f"Shapiro-Wilk Test:")
print(f"Test Statistic: {shapiro_test_stat:.4f}")
print(f"P-value: {shapiro_p_value:.4f}")

# Kolmogorov-Smirnov
ks_stat, ks_p_value = stats.kstest(y_pred_test, 'norm')
print(f"\nKolmogorov-Smirnov Test:")
print(f"Test Statistic: {ks_stat:.4f}")
print(f"P-value: {ks_p_value:.4f}")


# 将预测的概率转换为二分类标签
threshold = 0.5
y_pred_class_pca = (y_pred_pca >= threshold).astype(int)
y_test_class_pca = (y_test_pca >= threshold).astype(int)

roc_auc_pca = roc_auc_score(y_test_class_pca, y_pred_pca)
accuracy_pca = accuracy_score(y_test_class_pca, y_pred_class_pca)
precision_pca = precision_score(y_test_class_pca, y_pred_class_pca)
recall_pca = recall_score(y_test_class_pca, y_pred_class_pca)
f1_pca = f1_score(y_test_class_pca, y_pred_class_pca)
print(f"PCA-ROC-AUC Score: {roc_auc_pca}")
print(f"PCA-Accuracy: {accuracy_pca}")
print(f"PCA-Precision: {precision_pca}")
print(f"PCA-Recall: {recall_pca}")
print(f"PCA-F1 Score: {f1_pca}")

fpr, tpr, _ = roc_curve(y_test_class_pca, y_pred_class_pca)
roc_auc_value_pca = auc(fpr, tpr)

# plot roc
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='#205072', lw=2, label=f'PCA ROC curve (area = {roc_auc_value_pca:.2f})')
plt.plot([0, 1], [0, 1], color='#DE3A35', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("E://table//PCA后预测表现ROC.png")


y_pred_testD=pd.DataFrame(y_pred_test)
y_pred_testD.to_csv("E://table//y_pre_test第四问.csv")

y_predD=pd.DataFrame(y_pred)
y_predD.to_csv("E://table//y_pre第三问.csv")

y_pred_pcaD=pd.DataFrame(y_pred_pca)
y_pred_pcaD.to_csv("E://table//y_pre_pca第三问.csv")