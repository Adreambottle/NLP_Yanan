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