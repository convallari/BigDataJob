#%%
import jieba
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

data = pd.read_excel('job_bigData.xlsx')
#%%
print(len(data))
print(data.count())

#%%
# 查看数据缺失值
print(data.isnull().sum())

# 查看缺失数据
print(data[data['工作经验'].isnull()])
# print(data[data['职位描述'].isnull()])

#%% 统计
print(data['学历要求'].value_counts())
print(data['工作经验'].value_counts())
print(data['薪资待遇'].value_counts())

#%%
# 处理薪资待遇列
data['薪资下限'] = data['薪资待遇'].str.extract(r'(\d+)-(\d+)').iloc[:, 0].astype(int)
data['薪资上限'] = data['薪资待遇'].str.extract(r'(\d+)-(\d+)').iloc[:, 1].astype(int)

#%%
# 薪资下限描述统计
print(data['薪资下限'].describe())
# 薪资上限描述统计
print(data['薪资上限'].describe())

# 展示薪资下限最低的5个样本
print(data.sort_values('薪资下限').head())
# 展示薪资上限最高的5个样本
print(data.sort_values('薪资上限', ascending=False).head())

#%%
# 增加平均薪资列
data['薪资平均'] = (data['薪资下限'] + data['薪资上限']) / 2
# 薪资平均描述统计
print(data['薪资平均'].describe())
#%%
plt.figure(figsize=(8, 6))
# 淡紫色直方图
plt.hist(data['薪资平均'], bins=20, color='thistle')
# 增加均值线
mean_value = data['薪资平均'].mean()
plt.axvline(mean_value, color='r', linestyle='--')
# 标注均值
plt.text(mean_value, plt.gca().get_ylim()[1] * 0.9, f'均值: {mean_value:.2f}', color='r')
# 增加中位数线
median_value = data['薪资平均'].median()
plt.axvline(median_value, color='g', linestyle='--')
# 标注中位数
plt.text(median_value, plt.gca().get_ylim()[1] * 0.8, f'中位数: {median_value:.2f}', color='g')

plt.xlabel('薪资')
plt.ylabel('频数')
plt.title('薪资分布')
# 保存
plt.savefig('./output/salary_avg.png')
plt.show()
#%%
# 绘制薪资下限、上限直方图
plt.figure(figsize=(8, 6))
plt.hist(data['薪资下限'], bins=20, color='skyblue')
plt.hist(data['薪资上限'], bins=20, color='lightcoral', alpha=0.7)
plt.xlabel('薪资')
plt.ylabel('频数')
plt.title('薪资下限、上限分布')
# 图例
plt.legend(['薪资下限', '薪资上限'])
# 保存
# plt.savefig('./output/salary.png')
plt.show()

#%%
from scipy import stats
plt.rcParams['axes.unicode_minus'] = False
# 薪资下限正态分布检验
print(stats.normaltest(data['薪资下限']))
# 薪资上限正态分布检验
print(stats.normaltest(data['薪资上限']))

# QQ图，将检验结果标注在图上
plt.figure(figsize=(8, 6))
stats.probplot(data['薪资下限'], dist='norm', plot=plt)
plt.text(0.1, 0.9, f'p-value: {stats.normaltest(data["薪资下限"])[1]}', transform=plt.gca().transAxes)
plt.title('薪资下限QQ图')
plt.savefig('./output/salary_qq.png')
plt.show()

plt.figure(figsize=(8, 6))
stats.probplot(data['薪资上限'], dist='norm', plot=plt)
plt.text(0.1, 0.9, f'p-value: {stats.normaltest(data["薪资上限"])[1]}', transform=plt.gca().transAxes)
plt.title('薪资上限QQ图')
plt.savefig('./output/salary_qq2.png')
plt.show()
#%%
# 长尾分布检验
print(stats.skewtest(data['薪资下限']))
print(stats.skewtest(data['薪资上限']))

#%%
# 绘制薪资下限散点图，左子图纵坐标不变，右子图纵坐标对数坐标
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.scatter(range(len(data)), data['薪资下限'], c='skyblue')
plt.xlabel('样本')
plt.ylabel('薪资下限')
plt.title('线性纵坐标')
plt.subplot(1, 2, 2)
plt.scatter(range(len(data)), data['薪资下限'], c='skyblue')
plt.yscale('log')
plt.xlabel('样本')
plt.title('对数纵坐标')
plt.tight_layout()
# plt.savefig('./output/l_salary_scatter.png')
plt.show()

#%%
# 薪资上限与下限作差画图
plt.figure(figsize=(8, 6))
plt.hist(data['薪资上限'] - data['薪资下限'], bins=20, color='skyblue')
plt.xlabel('薪资上限与下限差值')
plt.ylabel('频数')
plt.title('薪资上限与下限差值分布')
# plt.savefig('./output/salary_diff.png')
plt.show()
#%%
# data删除工作经验缺失行
data.dropna(subset=['工作经验'], inplace=True)

# 写入excel
data.to_excel('job_bigData_cleaned.xlsx', index=False)
# data.to_csv('job_bigData_cleaned.csv', encoding='utf-8')

#%%
# # data删除职位描述列
# data.drop(columns=['职位描述'], inplace=True)
# #%%
# #data保存为csv，编码为utf-8
# data.to_csv('job_bigData.csv', encoding='utf-8')
#%%
# 薪资待遇统计创建dataframe
salary = data['薪资待遇'].value_counts().reset_index()
# salary提取的薪资待遇统计
salary['薪资下限'] = salary['薪资待遇'].str.extract(r'(\d+)-(\d+)').iloc[:, 0].astype(int)
salary['薪资上限'] = salary['薪资待遇'].str.extract(r'(\d+)-(\d+)').iloc[:, 1].astype(int)
# 绘制薪资待遇气泡图，横轴为薪资下限，纵轴为薪资上限，气泡大小颜色为数量，灰色背景
plt.scatter(salary['薪资下限'], salary['薪资上限'], s=salary['count'], c=salary['count'], cmap='cool')
plt.grid(True, linestyle='--', alpha=0.5)

plt.xlim(0,125000)
plt.ylim(0,125000)

plt.xlabel('薪资下限')
plt.ylabel('薪资上限')
plt.colorbar()
plt.title('薪资待遇气泡图')
plt.gcf().set_size_inches(6, 5)
# 保存
plt.savefig('./output/salary_bubble.png')
plt.show()
#%%
# 绘制学历要求条形图
edu_seq = ['不限', '大专', '本科', '硕士', '博士']
edu = data['学历要求'].value_counts()[edu_seq]

plt.figure(figsize=(8, 6))
edu.plot(kind='bar')
plt.title('学历要求统计')
plt.xlabel('学历要求')
plt.ylabel('计数')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./output/edu_bar.png')
plt.show()

#%%
# 不同学历的薪资待遇小提琴图，对于每类学历，左边画薪资下限，右边画薪资上限
lower_salary_color = 'skyblue'
upper_salary_color = 'lightcoral'

plt.figure(figsize=(8, 6))

for i in edu_seq:
    index = edu_seq.index(i) * 2


    violin1 = plt.violinplot(
        data[data['学历要求'] == i]['薪资下限'],
        positions=[index - 0.3],
        widths=0.6,
        showmeans=False,
        showmedians=True,
        showextrema=True
    )
    # 设置薪资下限的颜色
    for pc in violin1['bodies']:
        pc.set_facecolor(lower_salary_color)
        pc.set_edgecolor('black')  # 设置边框颜色
        pc.set_alpha(0.7)         # 设置透明度
    violin1['cmedians'].set_color('black')  # 设置中位数线颜色

    # 绘制薪资上限的小提琴图
    violin2 = plt.violinplot(
        data[data['学历要求'] == i]['薪资上限'],
        positions=[index + 0.3],
        widths=0.6,
        showmeans=False,
        showmedians=True,
        showextrema=True
    )
    # 设置薪资上限的颜色
    for pc in violin2['bodies']:
        pc.set_facecolor(upper_salary_color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    violin2['cmedians'].set_color('black')

# 设置 x 轴刻度
plt.xticks([i * 2 for i in range(len(edu_seq))], edu_seq)
plt.xlabel('学历要求')
plt.ylabel('薪资')
plt.title('不同学历的薪资待遇小提琴图')

# 添加图例
plt.legend(
    handles=[
        plt.Line2D([0], [0], color=lower_salary_color, lw=4, label='薪资下限'),
        plt.Line2D([0], [0], color=upper_salary_color, lw=4, label='薪资上限')
    ],
    loc='upper left'
)

plt.savefig('./output/salary_violin.png')
plt.show()

#%%
# 绘制工作经验条形图
workExp_seq = ['不限', '1-3年', '3-5年', '5-10年', '10年以上']
workExp = data['工作经验'].value_counts()[workExp_seq]

# 绘制条形图
plt.figure(figsize=(8, 6))
workExp.plot(kind='bar')
plt.title('工作经验要求统计')
plt.xlabel('工作经验要求')
plt.ylabel('计数')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./output/workExp_bar.png')
plt.show()

#%%
# 不同工作经验的薪资待遇小提琴图，对于每类工作经验，左边画薪资下限，右边画薪资上限
lower_salary_color = 'skyblue'
upper_salary_color = 'lightcoral'
plt.figure(figsize=(8, 6))
for i in workExp_seq:
    index = workExp_seq.index(i) * 2
    # 绘制薪资下限的小提琴图
    violin1 = plt.violinplot(
        data[data['工作经验'] == i]['薪资下限'],
        positions=[index - 0.3],
        widths=0.6,
        showmeans=False,
        showmedians=True,
        showextrema=True
    )
    # 设置薪资下限的颜色
    for pc in violin1['bodies']:
        pc.set_facecolor(lower_salary_color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    violin1['cmedians'].set_color('black')
    # 绘制薪资上限的小提琴图
    violin2 = plt.violinplot(
        data[data['工作经验'] == i]['薪资上限'],
        positions=[index + 0.3],
        widths=0.6,
        showmeans=False,
        showmedians=True,
        showextrema=True
    )
    # 设置薪资上限的颜色
    for pc in violin2['bodies']:
        pc.set_facecolor(upper_salary_color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    violin2['cmedians'].set_color('black')
# 设置 x 轴刻度
plt.xticks([i * 2 for i in range(len(workExp_seq))], workExp_seq)
plt.xlabel('工作经验要求')
plt.ylabel('薪资')
plt.title('不同工作经验的薪资待遇小提琴图')
# 添加图例
plt.legend(
    handles=[
        plt.Line2D([0], [0], color=lower_salary_color, lw=4, label='薪资下限'),
        plt.Line2D([0], [0], color=upper_salary_color, lw=4, label='薪资上限')
    ],
    loc='upper left'
)
plt.savefig('./output/salary_violin2.png')
plt.show()


#%%
# 绘制不同学历、工作经验的薪资三维图，横轴为学历，纵轴为工作经验，高度为薪资下限
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
edu_indices = data['学历要求'].apply(lambda x: edu_seq.index(x))
workExp_indices = data['工作经验'].apply(lambda x: workExp_seq.index(x))
# 绘制三维图
ax.bar3d(
    edu_indices,
    workExp_indices,
    data['薪资下限'],
    0.2,
    0.2,
    data['薪资下限'],
    shade=True
)
# 设置坐标轴标签
ax.set_xlabel('学历要求')
ax.set_ylabel('工作经验')
ax.set_zlabel('薪资下限')
plt.xticks(range(len(edu_seq)), edu_seq)
plt.yticks(range(len(workExp_seq)), workExp_seq)
plt.title('不同学历、工作经验的薪资三维图')
plt.show()

#%%
def get_cut_words(content_series):
    # 读入停用词表
    stop_words = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10']

    # with open(r"stop_words.txt", 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         stop_words.append(line.strip())

    # 添加关键词
    # my_words = []
    # for i in my_words:
    #     jieba.add_word(i)

    # 自定义停用词
    my_stop_words = ['以上','任职','以上学历','要求','本科','进行','包括','负责','相关','数据','技术','参与','问题','优先'
                     ]
    stop_words.extend(my_stop_words)

    # 分词
    word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)

    # 条件筛选
    word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2]

    return word_num_selected

# doc = pd.Series(['任职要求：\\t\\n1、计算机相关专业全日制本科及以上学历，8年及以上数据类(数仓、BI可视化、移动类数据展现等)开发经验，技术问题解决等能力包含:数仓调优SQL Server、数据可视化(PowerBI、Echart、dashboard)、 H5 移动端数据展示及高并发的实时数据处理经验作为加分，有ssis经验。\\n2、要求有一定研发/架构经验，理解Scrum理念，灵活运用 Scrum、看板两种方法\\t \\n3、良好的逻辑思维、业务解读能力，有独立分析解决问题的能力，责任心强，掌握协作，解决冲突，小组沟通的技巧;\\n4、具备优秀的项目管理、资源管理能力及领导力，具有PMP/ACP证书或相关项目管理证书及大型敏捷项目实践经验;\\n5、出色的时间管理，尤其是根据业务目标有效指定任务优先级及执行任务的能力\\n6、有金融公司实施项目经验者优先考虑\\n岗位职责:\\n1、领导和管理项目团队，负责项目的整体需求分析、设计指定及开发;\\n2、负责项目的分析、设计、开发、测试、交付，确保项目成功:\\n3、与客户相关沟通\\n4、负责项目技术方案设计，协调处理项目相关大的技术难点问题，一定的开发工作。\\n5、进行项目计划、工作统筹、任务分配、指定项目计划，量化任务，并合理分配给相关人员及任务的监控，并对其工作定期检查、评估。'], name='职位描述')
# print(get_cut_words(doc))
#%%
text = get_cut_words(content_series=data['职位描述'])
text[:10]

from PIL import Image
from wordcloud import WordCloud
import numpy as np

img = Image.open('cloudimage.png') #打开图片
img_array = np.array(img) #将图片装换为数组

wc = WordCloud(
    background_color='white',
    width=600,
    height=800,
    mask=img_array, #设置背景图片
    font_path=r'C:\Windows\Fonts\STKAITI.TTF', #设置字体
)
text_str = ' '.join(text)
wc.generate_from_text(text_str)#绘制图片
plt.imshow(wc)
plt.axis('off')#隐藏坐标轴
# wc.to_file('job_desc3.png')  #保存图片
# plt.savefig("job_desc3.png", dpi=800)
plt.show()  #显示图片