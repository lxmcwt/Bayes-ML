import jieba.analyse  # 导入分词库
import numpy as np  # 导入numpy库
import pandas as pd  # 导入pandas库
from sklearn.feature_extraction.text import TfidfVectorizer  # 导入求解TF-IDF值的SKlearn库
import csv  # 导入csv库
from functools import reduce  # 导入reduce，方便连乘条件概率

jieba.setLogLevel(jieba.logging.INFO)  # 该声明可以避免分词时报错
with open('stopwords.txt', 'r', encoding='utf-8') as file1:  # 打开停用词文件
    stopwords = file1.read()
    stopwords_list = stopwords.split('\n')  # 去除该txt文件里的换行
main_content = pd.read_csv('index_dic111.csv',
                           encoding='gbk')  # 读入上一部分导出的邮件正文对应标注好的文件
mail_text = main_content['main_content']  # 提取邮件正文列
cutwords_list = []  # 建立一个分词空列表
for line in mail_text:  # 对这个每行邮件正文迭代
    cutwords = [i for i in jieba.lcut(line) if i not in set(stopwords_list)]  # 每行正文分词并且剔除停用词
    cutwords_list.append(cutwords)  # 分好的一组词加入到分词列表
cutwords_list_deal = [' '.join(text) for text in cutwords_list]  # 将分词列表里的词用空格相连
"""开始计算每行邮件正文分词的TF-IDF值"""
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", max_features=7000, max_df=0.6, min_df=5)
# 剔除非中文字符，最大特征词数量控制在7000，同时在所有邮件中出现超过60%的不要（无代表性），也剔除仅出现小于5条的词汇
wrod_tfidf_array1 = tfidf.fit_transform(cutwords_list_deal)  # 将特征词的TF-IDF值转换为稀疏矩阵
keyword = tfidf.get_feature_names_out()  # 获取特征词的名称
wrod_tfidf_array2 = pd.DataFrame(wrod_tfidf_array1.toarray(), columns=keyword)
# 将特征词的名称，对应其TF-IDF值建立一个矩阵

print(wrod_tfidf_array1)  # 打印TF-IDF稀疏矩阵
print(wrod_tfidf_array2)  # 打印包含特征词的矩阵
print(tfidf.vocabulary_)  # 打印特征词

"""将特征词和序号导出一个表，方便自己看"""
with open("keyword_number.csv", "w", encoding="gbk", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["keyword", "number"])
    for keyword, number in tfidf.vocabulary_.items():
        csv_writer.writerow([keyword, number])
    print("写入数据成功")
    f.close()
"""将每行邮件的特征词导出来，无可厚非，可以看看"""
with open("cutwords_list_deal.csv", "w", encoding="gbk", newline="") as f:
    csv_writer = csv.writer(f)
    for cutwords in cutwords_list_deal:
        csv_writer.writerow([cutwords])
    print("写入数据成功")
    f.close()

"""切割好训练集，此时来计算每个词组的先验条件概率"""
label = main_content['label']  # 把标注列提取出来
wrod_tfidf_array2.insert(0, 'label', label)  # 加入到TF-IDF矩阵中的第一列前
word_tfidf_array2_train = wrod_tfidf_array2.iloc[:int(wrod_tfidf_array2.shape[0] * 0.7), :]
# 选取前70%的数据作为训练集
deal1 = word_tfidf_array2_train.groupby('label')  # 用groupby分组统计
wrod_tfidf_array2_train_deal = pd.DataFrame(deal1.sum())  # 分别给各个特征词的在标注为1和0时把TF-IDF值求和
spam_health_situation = pd.DataFrame(deal1.size())  # 看下训练集里垃圾邮件和健康邮件数量
print(spam_health_situation.iloc[0, 0])  # 看下健康邮件数量
print(spam_health_situation.iloc[1, 0])  # 看下垃圾邮件数量
"""下面注意正则化，防止0概率的出现影响结果"""
health_tfidf_sum = sum(wrod_tfidf_array2_train_deal.iloc[0]) + wrod_tfidf_array2_train_deal.shape[1]
spam_tfidf_sum = sum(wrod_tfidf_array2_train_deal.iloc[1]) + wrod_tfidf_array2_train_deal.shape[1]
# 分别求和垃圾和健康邮件的TF-IDF值，并且都加上正则化值，这里取的是列数，因为后续分母上每个特征词的TF-IDF值都要加1
list_cp0, list_cp1 = [], []  # 建立两个条件概率列表
for i in range(wrod_tfidf_array2_train_deal.shape[1]):  # 每列做迭代
    conditional_probablity_0 = (wrod_tfidf_array2_train_deal.iloc[0, i] + 1) / (health_tfidf_sum)
    conditional_probablity_1 = (wrod_tfidf_array2_train_deal.iloc[1, i] + 1) / (spam_tfidf_sum)
    list_cp0.append(conditional_probablity_0)
    list_cp1.append(conditional_probablity_1)
# 上述为每个元素都求好先验条件概率并加入健康和垃圾列表
for j in range(wrod_tfidf_array2_train_deal.shape[1]):
    wrod_tfidf_array2_train_deal.iloc[0, j] = list_cp0[j]
    wrod_tfidf_array2_train_deal.iloc[1, j] = list_cp1[j]
# 上述为再把两行TF-IDF求和表改变成条件概率列表，这里已经正则化

"""把训练集的条件概率列表导出来留作备用，当然不导出CSV也行"""
wrod_tfidf_array2_train_deal.to_csv('wrod_tfidf_array2_train_deal.csv')
print(wrod_tfidf_array2_train_deal)

"""下面开始测试数据"""
probablity_health_tem = []  # 建立一个健康条件概率临时表
probablity_spam_tem = []  # 建立一个垃圾条件概率临时表
compute_y_list = []  # 建立一个最终预测值表
word_tfidf_array2_test = wrod_tfidf_array2.iloc[int(wrod_tfidf_array2.shape[0] * 0.7) + 1:, :]
# 取出后30%数据作为测试集

for i in range(0, word_tfidf_array2_test.shape[0]):  # 外面对每行测试数据迭代
    for j in range(0, word_tfidf_array2_test.shape[1] - 1):  # 里面迭代每一列
        if word_tfidf_array2_test.iloc[i, j + 1] != 0:  # 当测试分词的TF-IDF值不为0时，即说明该词在这封邮件里有代表性
            probablity_health_tem.append(wrod_tfidf_array2_train_deal.iloc[0, j])
            # 健康列表加入对应到上面的条件概率表对应位置的概率
            probablity_spam_tem.append(wrod_tfidf_array2_train_deal.iloc[1, j])
            # 垃圾列表加入对应到上面的条件概率表对应位置的概率
        else:
            pass

    multiply_health = reduce(lambda x, y: x * y, probablity_health_tem, 1) * \
                      (spam_health_situation.iloc[0, 0] /
                       (spam_health_situation.iloc[0, 0] + spam_health_situation.iloc[1, 0]))
    # 计算这一封邮件为健康邮件时的后验条件概率
    multiply_spam = reduce(lambda x, y: x * y, probablity_spam_tem, 1) * \
                    (spam_health_situation.iloc[1, 0] /
                     (spam_health_situation.iloc[0, 0] + spam_health_situation.iloc[1, 0]))
    # 计算这一封邮件为垃圾邮件时的后验条件概率
    if multiply_spam >= multiply_health:  # 一个简单的判断
        y = 1
    else:
        y = 0
    compute_y_list.append(y)  # 预测值加入预测列表
    probablity_spam_tem.clear()
    probablity_health_tem.clear()  # 两个中转列表清空准备下一次迭代

"""可以计算各种误差值了"""
compute_y = np.array(compute_y_list)
real_y = np.array(word_tfidf_array2_test['label'])  # 将预测列表和真实列表值转换为数组方便运算
error_test = sum(compute_y - real_y) / len(compute_y)  # 计算测试误差

TP, FN, FP, TN = 0, 0, 0, 0  # 开始计算分类问题的四项指标
for i in range(0, len(compute_y)):
    if real_y[i] == 1 and compute_y[i] == 1:
        TP += 1
    if real_y[i] == 1 and compute_y[i] == 0:
        FN += 1
    if real_y[i] == 0 and compute_y[i] == 1:
        FP += 1
    if real_y[i] == 0 and compute_y[i] == 0:
        TN += 1
accurate_rate = TP / (TP + FP)  # 精确率
recall_rate = TP / (TP + FN)  # 召回率
correct_rate = (TP + TN) / (TP + TN + FP + FN)  # 准确率
F1_score = (2 * TP) / (2 * TP + FP + FN)  # F1值

print(f"误差率为：{error_test}")
print(f"准确率为：{correct_rate}")
print(f"精确率为：{accurate_rate}")
print(f"召回率为：{recall_rate}")
print(f"F1值为：{F1_score}")