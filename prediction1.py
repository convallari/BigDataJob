#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import alpha

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

data = pd.read_csv('job_bigData_cleaned.csv')

# 学历要求、工作经验标签编码
edu_mapping = {'大专': 1, '不限': 2, '本科': 3, '硕士': 4, '博士': 5 }

exp_mapping = {'1-3年': 1, '3-5年': 2,  '不限': 3, '5-10年': 4, '10年以上': 5 }

data['学历要求'] = data['学历要求'].map(edu_mapping)
data['工作经验'] = data['工作经验'].map(exp_mapping)

#写入文件
# data.to_csv('job_bigData_coded.csv', encoding='utf-8')

#%%
# 随机森林预测薪资
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

features = data[['学历要求', '工作经验']]
target = data['薪资下限']
# target = data['薪资上限']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# 计算 MAPE
mape = np.mean(abs(y_test - y_pred) / y_test)
print(f'MAPE: {mape}') #0.36

#%%
# 对于每个样本，绘制真实值（蓝色）点与预测值（红色）点，横坐标为样本，纵坐标为薪资
plt.scatter(range(len(y_test)), y_test, c='b', label='真实薪资')
plt.scatter(range(len(y_pred)), y_pred, c='r', label='预测薪资')
plt.xlabel('样本')
plt.legend()
plt.title('薪资预测结果')
# plt.savefig('./output/pre1_salary_pred.png')
plt.show()

# #%%
# # 筛选误差小的样本
# error = y_test - y_pred
# error = abs(error)
# error = error[error < 1000]
# print(error)
# # 展示这些样本的data数据
# temp = data.iloc[error.index]

#%%
# 合并features和target和预测
data_for_m = pd.concat([features, target,pd.Series(y_pred, index=y_test.index)], axis=1)
# 还原学历要求、工作经验标签
data_for_m['学历要求'] = data_for_m['学历要求'].map({v: k for k, v in edu_mapping.items()})
data_for_m['工作经验'] = data_for_m['工作经验'].map({v: k for k, v in exp_mapping.items()})
# 展示不同类别的误差
error = y_test - y_pred
error = abs(error)
data_for_m['误差'] = error
# 画图
plt.figure(figsize=(10, 6))
data_for_m.boxplot(column='误差', by=['学历要求', '工作经验'])
plt.xticks(rotation=45)
plt.title('不同学历、工作经验的薪资预测误差')
plt.suptitle('')
plt.tight_layout()
# plt.savefig('./output/pre1_error_boxplot.png')
plt.show()

#%%
import matplotlib.pyplot as plt
import pandas as pd

# 计算学历要求和工作经验组合的样本数量
grouped_data = data_for_m.groupby(['学历要求', '工作经验']).size().reset_index(name='样本数量')

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制样本数量的条形图
ax1.bar(grouped_data.index, grouped_data['样本数量'], tick_label=grouped_data.apply(lambda row: f"{row['学历要求']}, {row['工作经验']}", axis=1), color='lightblue',alpha = 0.7)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_xlabel('学历要求与工作经验组合')
ax1.set_ylabel('样本数量', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# 图例
plt.legend(['样本数量'], loc='upper left')
# 创建第二个坐标轴（共享 x 轴）
ax2 = ax1.twinx()

# 绘制箱形图
data_for_m.boxplot(column='误差', by=['学历要求', '工作经验'], ax=ax2, boxprops=dict(color='lightcoral'), whiskerprops=dict(color='lightcoral'), capprops=dict(color='lightcoral'), flierprops=dict(color='lightcoral',  markeredgecolor='lightcoral'), medianprops=dict(color='r'))
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.set_ylabel('误差', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# 设置图表标题和布局
plt.title('不同学历、工作经验的薪资预测误差与样本数量')
plt.suptitle('')
plt.tight_layout()

# 图例
plt.legend(['误差'], loc='upper right')
# 保存图表
plt.savefig('./output/pre1_error_boxplot2.png')
plt.show()
