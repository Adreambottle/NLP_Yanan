import jieba
import jieba.posseg as pseg     # 词性标注
import jieba.analyse as anls    # 关键词提取

text = "我毕业于香港中文大学数理统计专业，有很强的数理和编程背景。"

# jieba.cut() 返回的是一个 generator，全模式
# cut_all 返回的是贪婪模式，包括所有中文词汇的提取，

seg_list = jieba.cut(text, cut_all=True)
print("/".join(seg_list))

# cut_all 返回的是精确模式，不会包括重复的中文词汇提取
seg_list = jieba.cut(text, cut_all=False)
print("/".join(seg_list))

# jieba.lcut() 和 jieba.cut() 是一样的，返回的是一个 list
seg_list = jieba.lcut(text, cut_all=True)
print(seg_list)

# jieba.cut_for_search
seg_list = jieba.cut_for_search(text)
print("/".join(seg_list))


# HMM 模式
"""
HMM 模型，即隐马尔可夫模型(Hidden Markov Model, HMM)
采用四个隐含状态，分别表示为单字成词，词组的开头，词组的中间，词组的结尾。
通过标注好的分词训练集，可以得到 HMM 的各个参数，
然后使用 Viterbi 算法来解释测试集，得到分词结果。

"""

# 未启用 HMM
seg_list = jieba.cut("他来到了网易杭研大厦", HMM=False) #默认精确模式和启用 HMM
print("【未启用HMM】:" + "/".join(seg_list))  


# 识别新词
seg_list = jieba.cut("他来到了网易杭研大厦") #默认精确模式和启用 HMM
print("【识别新词】：" + "/ ".join(seg_list))  

# 使用 jieba.load_userdict(file_name) 即可载入词典。
sample_text = "周大福是创新办主任也是云计算方面的专家"

# 未加载词典
print("【未加载词典】：" + '/'.join(jieba.cut(sample_text)))

jieba.add_word('石墨烯') #增加自定义词语
jieba.add_word('凱特琳', freq=42, tag='nz') #设置词频和词性 
jieba.del_word('自定义词') #删除自定义词语 

# 关键词提取
# TF-IDF(Term Frequency-Inverse Document Frequency, 词频-逆文件频率)是一种统计方法，用以评估一个词语对于一个文件集或一个语料库中的一份文件的重要程度，其原理可概括为：
# 一个词语在一篇文章中出现次数越多，同时在所有文档中出现次数越少，越能够代表该文章
s = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
for x, w in anls.extract_tags(s, topK=20, withWeight=True):
    print('%s %s' % (x, w))


# 基于 TextRank 算法的关键词提取
for x, w in anls.textrank(s, withWeight=True):
    print('%s %s' % (x, w))