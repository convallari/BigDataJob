#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

data = pd.read_csv('job_bigData_coded.csv')

data['职位描述'].fillna('', inplace=True) #否则中文分词会报错

#%%
# 画图查看职位描述列的长度分布
plt.figure(figsize=(10, 6))
plt.hist(data['职位描述'].str.len(), bins=50, color='skyblue')
plt.xlabel('职位描述长度')
plt.ylabel('频数')
plt.title('职位描述长度分布')
# plt.savefig('./output/pre2_job_desc_len.png')
plt.show()

#%%
# 查看职位描述长度低于100的data
data[(data['职位描述'].str.len() > 10) & (data['职位描述'].str.len() < 100)]['职位描述']

#%%

import jieba
from gensim.utils import simple_preprocess

# 中文分词函数
def chinese_tokenizer(text):
    # 添加自定义词典
    my_words = ['数仓']
    for word in my_words:
        jieba.add_word(word)

    return jieba.lcut(text)

# 英文分词函数（英文按词拆分）
def english_tokenizer(text):
    return simple_preprocess(text)

# 分词函数，自动检测中文和英文
def mixed_tokenizer(text):
    # 分开中文和英文
    chinese_part = ''.join([char for char in text if '\u4e00' <= char <= '\u9fff'])  # 只保留中文字符
    english_part = ''.join([char for char in text if not ('\u4e00' <= char <= '\u9fff')])  # 只保留英文字符

    chinese_tokens = chinese_tokenizer(chinese_part) if chinese_part else []
    english_tokens = english_tokenizer(english_part) if english_part else []

    return chinese_tokens + english_tokens

# 应用混合分词
data['分词'] = data['职位描述'].apply(mixed_tokenizer)

# doc = ['任职要求：\\t\\n1、计算机相关专业全日制本科及以上学历，8年及以上数据类(数仓、BI可视化、移动类数据展现等)开发经验，技术问题解决等能力包含:数仓调优SQL Server、数据可视化(PowerBI、Echart、dashboard)、 H5 移动端数据展示及高并发的实时数据处理经验作为加分，有ssis经验。\\n2、要求有一定研发/架构经验，理解Scrum理念，灵活运用 Scrum、看板两种方法\\t \\n3、良好的逻辑思维、业务解读能力，有独立分析解决问题的能力，责任心强，掌握协作，解决冲突，小组沟通的技巧;\\n4、具备优秀的项目管理、资源管理能力及领导力，具有PMP/ACP证书或相关项目管理证书及大型敏捷项目实践经验;\\n5、出色的时间管理，尤其是根据业务目标有效指定任务优先级及执行任务的能力\\n6、有金融公司实施项目经验者优先考虑\\n岗位职责:\\n1、领导和管理项目团队，负责项目的整体需求分析、设计指定及开发;\\n2、负责项目的分析、设计、开发、测试、交付，确保项目成功:\\n3、与客户相关沟通\\n4、负责项目技术方案设计，协调处理项目相关大的技术难点问题，一定的开发工作。\\n5、进行项目计划、工作统筹、任务分配、指定项目计划，量化任务，并合理分配给相关人员及任务的监控，并对其工作定期检查、评估。']
# print(mixed_tokenizer(doc[0]))

#%%
from gensim.models import Word2Vec

# 训练 Word2Vec 模型
model = Word2Vec(
    sentences=data['分词'],  # 使用“分词”列作为输入
    vector_size=100,         # 词向量维度
    window=5,                # 上下文窗口大小
    min_count=1,             # 最小词频
    workers=4,               # 线程数
    sg=0                    # CBOW 模型
)

# 将句子转换为向量（对所有词的词向量取平均）
def sentence_to_vector(sentence, model):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# 将每个职位描述转换为向量
data['职位描述向量'] = data['分词'].apply(lambda x: sentence_to_vector(x, model))

# 降维
from sklearn.decomposition import PCA
# 获取向量数组
vectors = np.array(data['职位描述向量'].tolist())
# 使用 PCA 将维度从 100 降到 10
pca = PCA(n_components=10)
vectors_10d = pca.fit_transform(vectors)
# 将降维后的向量保存回 DataFrame
data['职位描述向量_10d'] = list(vectors_10d)


#%%
# 将公司名称转换为向量
data['公司名称分词'] = data['公司名称'].apply(mixed_tokenizer)
model = Word2Vec(
    sentences=data['公司名称分词'],  # 使用“分词”列作为输入
    vector_size=100,         # 词向量维度
    window=5,                # 上下文窗口大小
    min_count=1,             # 最小词频
    workers=4,               # 线程数
    sg=0                     # CBOW 模型
)
data['公司名称向量'] = data['公司名称分词'].apply(lambda x: sentence_to_vector(x, model))
# 降维
from sklearn.decomposition import PCA
# 获取向量数组
vectors = np.array(data['公司名称向量'].tolist())
# 使用 PCA 将维度从 100 降到 2
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)
# 将降维后的向量保存回 DataFrame
data['公司名称向量_2d'] = list(vectors_2d)

#%%
# 将招聘岗位转换为向量
data['招聘岗位分词'] = data['招聘岗位'].apply(mixed_tokenizer)
model = Word2Vec(
    sentences=data['招聘岗位分词'],  # 使用“分词”列作为输入
    vector_size=100,         # 词向量维度
    window=5,                # 上下文窗口大小
    min_count=1,             # 最小词频
    workers=4,               # 线程数
    sg=0                     # CBOW 模型
)

data['招聘岗位向量'] = data['招聘岗位分词'].apply(lambda x: sentence_to_vector(x, model))

# 降维
from sklearn.decomposition import PCA
# 获取向量数组
vectors = np.array(data['招聘岗位向量'].tolist())
# 使用 PCA 将维度从 100 降到 2
pca = PCA(n_components=10)
vectors_2d = pca.fit_transform(vectors)
# 将降维后的向量保存回 DataFrame
data['招聘岗位向量_2d'] = list(vectors_2d)


#%%
# 将职位描述向量与学历要求和工作经验合并
def combine_features(row):

    # 将其他特征（学历要求、工作经验）与职位描述向量拼接
    combined_vector = np.concatenate([row['职位描述向量_10d'], [row['学历要求']], [row['工作经验']],
                                      row['公司名称向量_2d'], row['招聘岗位向量_2d'] ])
    return combined_vector

# 为每一行数据生成合并后的特征向量
data['综合特征'] = data.apply(lambda row: combine_features(row), axis=1)

#%%
# 综合特征列降维
from sklearn.decomposition import PCA
# 获取向量数组
vectors = np.array(data['综合特征'].tolist())
# 使用 PCA 将维度降到 10
pca = PCA(n_components=10)
vectors_10d = pca.fit_transform(vectors)
# 将降维后的向量保存回 DataFrame
data['综合特征_10d'] = list(vectors_10d)

#%%
# 将data的综合特征列写入csv，根据综合特征的向量维度确定写入的列数
import numpy as np
import pandas as pd

# 计算综合特征向量的维度
vector_dim = len(data['综合特征_10d'][0])  # 获取第一个综合特征向量的维度

# 生成列名，第一列为“职位描述”，之后为向量维度列
columns = [f'f_{i+1}' for i in range(vector_dim)]

df_features = pd.DataFrame(columns=columns)

# 将综合特征向量展开为每个维度对应的列
for i in range(vector_dim):
    df_features[f'f_{i+1}'] = [row[i] for row in data['综合特征_10d']]

# 合并薪资列
df_features['lsalary'] = data['薪资下限']
df_features['hsalary'] = data['薪资上限']

# 保存到 csv文件
# df_features.to_csv('combined_features.csv', index=False)

print(df_features.head())

#%%
# 训练RandomForestRegressor模型
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
clf = RandomForestRegressor(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# 预测
y_pred = clf.predict(X_test)
# 计算MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}') # 0.34
