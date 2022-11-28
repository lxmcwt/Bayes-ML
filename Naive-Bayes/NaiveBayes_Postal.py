import codecs  # 解码器
import os  # 文件执行模块
import re  # 字符串匹配库
import csv

path = 'trec06c\data'

"""将所有邮件中的正文部分剔除掉所有非中文字符和空格并提取出来"""
groups = os.listdir(path)  # 返回一个路径下的所有目录名
main_content_dic = {}  # 创建一个空字典，留着存储邮件编号和对应的正文内容
for group in groups:  # 迭代子目录
    group_path = path + '\\' + group  # 将文件路径再深入一级
    mail_number = os.listdir(group_path)  # 再返回深入一级的文件目录
    for mail in mail_number:  # 迭代邮件文件目录
        mail_path = group_path + '\\' + mail  # 定义根目录的路径
        key = str(mail_path)[-8:-4] + str(mail_path)[-3:]  # 作为字典的键
        mail_content = codecs.open(mail_path, 'r', 'gbk', errors='ignore')  # 打开每个邮件
        count = 1  # 记录初始行数为1
        """下面为了定位到正文开始的那一行"""
        for line in mail_content.readlines():  # 逐行读取这封邮件
            line_noblank = line.rstrip()  # 剔除空行
            if line_noblank != '':  # 搞个条件判断，如果该行不为空，那么行数就加1
                count += 1
            else:
                break  # 退出的条件就是该行为空了，也就是邮件正文内容了
        mail_content.seek(0)  # 指针返回开头
        rows = len(mail_content.readlines())  # 统计该邮件总行数
        mail_content.seek(0)  # 指针再返回开头
        main_content = mail_content.readlines()[count:rows]  # 只读取有正文开始前的空白行到邮件末尾
        email = ''  # 定义一个空字符串用来存邮件正文
        """这时候才是返回出正文并生成字典"""
        for line in main_content:  # 再次迭代
            line_noblank = line.strip('\n')  # 再次将正文中的空行剔除
            email += line_noblank  # 给加到空字符串中
        email_noblank = email.replace(" ", "")  # 剔除掉正文中所有空格
        email_noblank = re.sub("([^\u4e00-\u9fa5])", '', email_noblank)  # 剔除掉正文中所有非中文字符
        main_content_dic[key] = email_noblank

"""将邮件的标签提取出来建立字典"""
index_content = codecs.open('trec06c\\full\\index',
                            'r', 'gbk', errors='ignore')  # 打开索引文件
index_list = []  # 定义空列表
index_dic = {}  # 定义空字典
for line in index_content.readlines():  # 逐行读取索引文件
    line_noblank = line.strip()  # 剔除空行
    line_sub = line_noblank.replace("/", '\\')  # 将每行里的斜杠替换成跟邮件地址序号一致的（方便后续匹配）
    index_list.append(line_sub)  # 列表加入一个该行内容
for i in index_list:  # 再迭代这个列表
    label = str(i)[:4]  # 将标签提取出
    if label == 'spam':  # 将是否为垃圾邮件定义为1或0
        label = 1
    else:
        label = 0
    index = str(i)[-8:-4] + str(i)[-3:]
    index_dic[index] = label  # 添加字典，将地址序号作为键，标签作为值

with open("test1\main_content_dic111.csv", "w", encoding="gbk", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["index", "main_content"])
    for index, main_content in main_content_dic.items():
        csv_writer.writerow([str(index), main_content])
    print("写入数据成功")
    f.close()
with open("test1\index_dic111.csv", "w", encoding="gbk", newline="") as g:
    csv_writer = csv.writer(g)
    csv_writer.writerow(["index", "label"])
    for index, label in index_dic.items():
        csv_writer.writerow([str(index), label])
    print("写入数据成功")
    g.close()
