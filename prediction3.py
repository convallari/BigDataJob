#%%
# 读csv文件
import pandas as pd
df = pd.read_csv('combined_features.csv')

# 划分数据集
from sklearn.model_selection import train_test_split
X = df.drop(columns=['lsalary', 'hsalary'])
y = df['lsalary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练Gradient Boosting Regression模型
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
clf.fit(X_train, y_train)
# 预测
y_pred = clf.predict(X_test)
# 计算MAPE
mape = np.mean(abs(y_test - y_pred) / y_test)
print(f'MAPE: {mape}') #0.30
