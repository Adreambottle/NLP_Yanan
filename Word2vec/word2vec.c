//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000            // EXP_TABLE_SIZE 代表 Exponention 中只存储 1000 个值，不再计算，只采用查表的方法
#define MAX_EXP 6                      // MAX_EXP 是一个常数，把横坐标从-6 到 6 等分为 1000 份，方便对 Exponential 值查表
#define MAX_SENTENCE_LENGTH 1000       // MAX_SENTENCE_LENGTH 代表每个句子的最大的长度是1000个单词
#define MAX_CODE_LENGTH 40             // MAX_CODE_LENGTH 是 Huffman Tree 的深度吗，也就是单词编码的长度

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
// const 后面定义的是常量，例如const int Max=100,则 Max++会报错

typedef float real;                    // 把 float 定义为 real，真的没有必要


// 定义每个词的基本结构类
struct vocab_word {
  long long cn;           // 词频，从训练集中计数或者直接提供词频文件
  int *point;             // Haffman树中从根节点到该词的路径，存放的是路径上每个节点的索引  
  char *word,             // word 是该词的字面值
       *code,             // code 是该词的 haffman 编码
       codelen;           // codelen 为该词haffman编码的长度
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];

// 词表，该数组的下标表示这个词在此表中的位置，也称之为这个词在词表中的索引
struct vocab_word *vocab;

int binary = 0, 
    cbow = 1, 
    debug_mode = 2, 
    window = 5, 
    min_count = 5, 
    num_threads = 12, 
    min_reduce = 1;

// 词hash表，该数组的下标为每个词的hash值，由词的字面ASCII码计算得到。vocab_hash[hash]中储存的是该词在词表中的索引
int *vocab_hash;


long long vocab_max_size = 1000,       // vocab_max_size 是一个辅助变量，每次当词表大小超出 vocab_max_size 时，一次性将词表大小增加1000
          vocab_size = 0,              // vocab_size为训练集中不同单词的个数，即词表的大小，在未构建词表时，大小当然为0
          layer1_size = 100;           // layer1_size为词向量的长度
long long train_words = 0, 
          word_count_actual = 0, 
          iter = 5, 
          file_size = 0, 
          classes = 0;
real alpha = 0.025, 
     starting_alpha, 
     sample = 1e-3;



                      // real是取实数的意思
real  *syn0,          // syn0存储的是词表中每个词的词向量
      *syn1,          // syn1存储的是Haffman树中每个非叶节点的向量
      *syn1neg,       // syn1neg是负采样时每个词的辅助向量
      *expTable;      // expTable是提前计算好的Sigmoid函数表

clock_t start;        //计时函数

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

//计算每个函数的能量分布表，在负采样中用到
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  //为能量表table分配内存空间，共有table_size项，table_size为一个既定的数1e8
  table = (int *)malloc(table_size * sizeof(int)); //malloc函数分配所需的内存空间
  //遍历词表，根据词频计算能量总值
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  //d1表示已遍历词的能量值占总能量的比
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  //a：能量表table的索引
  //i：词表的索引
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    //i号单词占据table中a的位置
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// 从文件中读入一个词到word，以space'',tab'\t',E0l'\n'为词的分界符
//截去一个词中长度超过MAX_STRING的部分
//每一行的末尾输入一个</s>
void ReadWord(char *word, FILE *fin)
//fin和fout都表示是文件流指针，即FILE*，用于读写文件fin这里用于读取in.txt，fout用于向文件out.txt写入数据
{
  int a = 0, ch;
  while (!feof(fin)) //当条件不是文件尾时均执行后续代码
  {
    ch = fgetc(fin);// fgetc指从文件指针指向的文件中读取一个字符，读取一个字节后，光标位置后移一个字节
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin); //ungetc把一个（或多个）字符退回到stream代表的文件流中，可以理解为一个“计数器”
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // 截去太长的单词
  }
  word[a] = 0;
}

// 返回一个词的hash值，由词的字面值计算得到，可能存在不同词拥有相同hash值的冲突情况
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// 返回一个词在词表中的位置，若不存在则返回-1
//先计算词的hash值，然后在词hash表中，以该值为下标，查看对应的值
//如果为-1说明这个词不存在索引，即不存在的词表中，返回1
//如果该索引在词表中对应的词与正在查找的词不符，说明hash值发生了冲突，按照开放地址法寻找这个单词。

int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

//从文件中读入一个词，并返回这个词在词表中的位置，相当于将之前的两个函数包装了起来
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// 为一个词构建一个vocab_word结构对象，并添加到词表中
//词频初始化为0，hash值用之前的函数计算
//返回该词在词表中的位置
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char)); //calloc与malloc类似，都是在存储区中分配指定长度的空间
  strcpy(vocab[vocab_size].word, word); // strcpy(char* des,const char* source),把从src地址开始且含有null结束符的字符串复制到以dest开始的地址空间
  vocab[vocab_size].cn = 0; // ??????????
  vocab_size++;
//每当词表数目即将超过最大值时，一次性为其申请添加一千个词结构体的内存空间
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  //如果该hash值与其他词产生冲突，则适用开放地址法解决冲突（为这个词寻找一个hash值空位）
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  //将该词在词表中的位置赋给这个找到的hash值空位
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// 按照词频从大到小排序
int VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// 统计词频，按照词频对词表中的项从大到小排序
void SortVocab() {
  int a, size;
  unsigned int hash;
  //对词表进行排序，将</s>放在第一个位置
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare); //qsort是C语言自带的排序函数
  //重置hash表
  for (a = 0; a < vocab_hash_size; a++)
    vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // 将出现次数小于min_count的词从词表中去除，出现次数大于min_count的重新计算hash值，更新hash词表
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word); //free是释放内存空间的函数
    } else {
      // 重新计算hash值
      hash = GetWordHash(vocab[a].word);
      // hash值冲突解决
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      // 计算总词数
      train_words += vocab[a].cn;
    }
  }
  // 由于删除了词频较低的词，这里调整词表的内存空间
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // 为Haffman树的构建预先申请空间
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}


// 从词表中删除出现次数小于min_reduce的词，每执行一次该函数min_reduce自动加一
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    } else free(vocab[a].word);
  vocab_size = b;
  // 重置hash表
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  // 更新hash表
  for (a = 0; a < vocab_size; a++) {
    // hash值计算
    hash = GetWordHash(vocab[a].word);
    // hash值冲突解决
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}


//         Root
//          /\
//      NL       NL
//     / \      / \
//  NL    NL    NL  g
//  /\    /\    /\   
// a  b  c  d  e  f

// 利用统计到的词频构建Haffman二叉树
// 按照Haffman树的特性，出现频率越高的词其二叉树上的路径越短，即二进制编码越短
// Huffman Tree 只有每个叶节点才表示一个单词，非叶节点只是 0 或 1，表示到达叶节点的路径
// Huffman 编码，每个词 w 都可以从树的根结点沿着唯一一条路径被访问到
// Huffman Tree 的构建是 bottom-up 的形式，从 N 个叶子节点代表的 N 个单词生成 (N-1) 个路径节点

void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2;     // 用来暂存一个词到根结点的Haffman树路径

  long long point[MAX_CODE_LENGTH];      // point[MAX_CODE_LENGTH] 是一个Array，用来暂存一个词的HAffman编码

  char code[MAX_CODE_LENGTH];            // code是什么

  // count数组前vocab_size个元素为Haffman树的叶子节点，初始化为词表中所有词的词频
  // count数组后vocab_size个元素为Haffman树中即将生成的非叶子节点（合并节点）的词频，初始化为一个大值1e15
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

  // binary数组记录各节点相对于其父节点的二进制编码（0/1）
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

  // parent 数组记录每个节点的父节点
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

  // count数组的初始化
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;           // 前vocab_size个元素为叶子节点，初始化为词表中所有词的词频
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;     // 后vocab_size个元素为非叶子节点，初始化为一个大值1e15

  // 以下部分为创建 Haffman 树的算法，默认词表已经按词频由高到低排序
  pos1 = vocab_size - 1;                       // pos1 为词表中词频倒数第二低的词下标（就是词表中倒数第二个的）
  pos2 = vocab_size;                           // pos2 为词表中词频最低的词的下标（就是词表最末尾的）
  
  // 最多进行vocab_size-1次循环操作，每次添加一个节点，即可构成完整的树
  for (a = 0; a < vocab_size - 1; a++) {
    // 比较当前的pos1和pos2，在min1i、min2i中记录当前词频最小和次小节点的索引
    // min1i和min2i可能是叶子节点也可能是合并后的中间节点
    if (pos1 >= 0) {
      //如果coun[pos1]比较小，则pos1左移，反之pos2右移
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      // 如果coun[pos1]比较小，则pos1左移，反之pos2右移
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    // 在count数组的后半段存储合并节点的词频（即最小count[min1i]和次小count[min2i]词频之和）
    count[vocab_size + a] = count[min1i] + count[min2i];
    // 记录min1i和min2i节点的父节点
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    // 这里令每个节点的左右子节点中，词频较低的为1（则词频较高的为0）
    binary[min2i] = 1;
  }
  // 根据得到的Haffman二叉树为每个词（树中的叶子节点）分配Haffman编码
  // 由于要为所有词分配编码，因此循环vocab_size次
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      
      // 不断向上寻找叶子结点的父节点，将binary数组中存储的路径的二进制编码增加到code数组末尾
      code[i] = binary[b];     // code是该词的haffman编码
      // 在point数组中增加路径节点的编号
      point[i] = b;
      // Haffman编码的当前长度，从叶子结点到当前节点的深度
      b = parent_node[b];
      // 由于Haffman 树共有vocab_size*2-1个节点，所以vocab_size*2-2为根节点
      i++;
      if (b == vocab_size * 2 - 2) break;
    }
    // 在词表中更新该词的信息
    // Haffman编码的长度，即叶子结点到根节点的深度
    vocab[a].codelen = i;
    // Haffman路径中存储的中间节点编号要在现在得到的基础上减去vocab_size，即不算叶子结点，单纯在中间节点中的编号
    // 所以现在根节点的编号为(vocab_size*2-2)-vocab_size=vocab_size-2
    vocab[a].point[0] = vocab_size - 2;
    // Haffman编码和路径都应该是从根节点到叶子结点的，因此需要对之前得到的code和point进行反向
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}



//从训练文件中获取所有词汇并构建词表和hash表
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;

  // 初始化hash词表
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  // 打开训练文件
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }

  //初始化词表大小
  vocab_size = 0;
  // 将</s>添加到词表的最前端
  AddWordToVocab((char *)"</s>");

  //开始处理训练文件
  while (1) {
    // 从文件中读入一个词
    ReadWord(word, fin);
    if (feof(fin)) break;
    // 对总词数加一，并输出当前训练信息
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    //搜索这个词在词表中的位置
    i = SearchVocab(word);
    //如果词表中不存在这个词，则将该词添加到词表中，创建其在hash表中的值，初始化词频为1；反之，词频加一
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    // 如果词表大小超过上限，则做一次词表删减操作，将当前词频最低的词删除
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  // 对词表进行排序，剔除词频低于阈值min_count的值，输出当前词表大小和总词数
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  // 获取训练文件的大小，关闭文件句柄
  file_size = ftell(fin);
  fclose(fin);
}

//将单词和对应的词频输出到文件中
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

// 从词汇表文件中读词并构建词表和hash表
// 由于词汇表中的词语不存在重复，因此与LearnVocabFromTrainFile相比没有做重复词汇的检测
void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  // 打开词汇表文件
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  // 初始化hash词表
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  // 开始处理词汇表文件
  while (1) {
    // 从文件中读入一个词
    ReadWord(word, fin);
    if (feof(fin)) break;
    // 将该词添加到词表中，创建其在hash表中的值，并通过输入的词汇表文件中的值来更新这个词的词频
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  // 对词表进行排序，剔除词频低于阈值min_count的值，输出当前词表大小和总词数
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  // 打开训练文件，将文件指针移至文件末尾，获取训练文件的大小
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  // 关闭文件句柄
  fclose(fin);
}


// syn0 存储的是词表中每个词的词向量，初始化为0
// syn1 储存的是 Haffman Tree 中每个非叶节点的向量，初始化为一个[-0.5, 0.5]的小数
// syn1neg 储存的是负采样样本的词向量，初始化为0
// syn0, syn1, syn1neg 都是一个 vocab_size * layer_size 的矩阵

//[[a0, a1, a2, ..., a100],            embedding 的维度即 layer1_size 是100
// [b0, b1, b2, ..., b100],            vocab_size 的维度，即词表的长度，这里用 a-z 26
// [c0, c1, c2, ..., c100],
//         ...
// [z0, z1, z2, ..., z100]]

//初始化神经网络结构
void InitNet() {
  
  long long a, b;   
  unsigned long long next_random = 1;

  // syn0存储的是词表中每个词的词向量, syn1储存的是Haffman Tree 中每个非叶节点的向量，这里为syn0分配内存空间
  // 调用 posiz_meanlign 来获取一块数量为 vocab_size*layer_size, 128byte 页对齐的内存，其中layer_size是词向量的长度
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}

  // 如果使用多层Softmax回归
  // 则需要为 syn1 分配内存空间。syn1存储的是Haffman树中每个非叶节点的向量。
  if (hs) {
    // syn1 分配内存空间
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    // syn1 初始化为0
    for (a = 0; a < vocab_size; a++) 
      for (b = 0; b < layer1_size; b++)
        syn1[a * layer1_size + b] = 0;    
  }

  // 如果要使用负采样
  // 则需要为 syn1neg 分配内存空间。syn1neg是负采样时每个词的辅助向量
  if (negative > 0) {
    
    // syn1neg 分配内存空间
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
        
    // syn1neg 初始化为0
    for (a = 0; a < vocab_size; a++) 
      for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] = 0;
  }

  // syn0 初始化
  // 每一维的值为[-0.5，0.5]/layer_size范围内的随机数
  for (a = 0; a < vocab_size; a++) 
    for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      // 生成 syn0 的随机数
      syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
  
  // 创建 Haffman 二叉树
  CreateBinaryTree();
}



// CBOW 和 Skipgram 都有两种可以选用的算法，Hierachical Softmax 和 Negative Sampling
// CBOW 删除了最耗时的非线性隐层，而且所有词都是共享隐藏等的

//该函数为线程函数，是训练算法代码实现的主要部分
//默认在执行该线程函数前，已经完成词表排序、Haffman树的生成以及每个词的Haffman编码计算
void *TrainModelThread(void *id) {

  long long a, b, d,
    cw,                       // cw:窗口长度（中心词除外）
    word,                     // word:在提取句子时用来表示当前词在词表中的索引
    last_word,                // last_word:用于在窗口扫描辅助，记录当前扫描到的上下文单词
    sentence_length = 0,      // sentence_length:当前处理的句子长度
    sentence_position = 0;    // sentence_position:当前处理的单词在当前句子中的位置

  long long
    word_count = 0,           // word_count: 当前线程当前时刻已经训练的语料的长度
    last_word_count = 0,      // last_word_count: 当前线程上一次记录时已训练的语料长度
    sen[MAX_SENTENCE_LENGTH + 1];     //sen:当前从文件中读取的待处理的句子，存放的是每个词在词表中的索引

  long long
    l1,                       // l1:在skip—gram模型中，在syn0中定位当前词词向量的起始位置
    l2,                       // l2:在syn1或syn1neg中定位中间节点向量或负采样向量的起始位置
    c,                        // 在滑动窗口操作的时候特指窗口内的某个单词
    target,                   // target:在负采样中存储当前样本
    label,                    // label:在负采样中存储当前样本的标记
    local_iter = iter;

  //next_random:用来辅助生成随机数
  unsigned long long
  next_random = (long long)id;
  real f, g;                     // real 就是 float 的意思
  clock_t now;
  
  // neu1:  输入词向量，在CBOW模型中是 Context(x) 即窗口内中各个词的向量和，在skip-gram模型中是中心词的词向量
  // neu1e: 累计的误差项，即 |real - pred|, neu1 和 neu1e 都是内存上连续100个float的向量，
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));     // layer1_size = 100，表示每个单词映射成100个维度
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  
  FILE *fi = fopen(train_file, "rb");
  
  //每个进程对应一段文本，根据当前线程的id找到该线程对应文本的初始位置
  //file_size就是之前LearnVocabFramTrainFile和ReadVocab函数中获取的训练文件的大小
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);

  //开始主循环
  while (1) {
    
    if (word_count - last_word_count > 10000) {             // 上次训练的文本量和本次训练的文本量，每训练约1000词输出一次训练进度
      
      word_count_actual += word_count - last_word_count;    // word_count_actual 是所有线程总共当前处理的词数
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        //输出信息包括:
        //当前的学习率alpha;
        //训练总进度（当前训练的总词数/（迭代次数*训练样本总词数）+1））;
        //每个线程每秒处理的词数
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
               word_count_actual / (real)(iter * train_words + 1) * 100,
               word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      //在初始学习率的基础上，随着实际训练词数的上升，逐步降低当前学习率（自适应调整学习率）
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      //调整的过程中保证学习率不低于starting_alpha*0.001
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    //从训练样本中取出一个句子，句子间以回车分割
    if (sentence_length == 0) {
      while (1) {
        //从文件中读入一个词，将该词在词表中的索引赋给word
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        //如果读到的是回车，表示句子结束
        if (word == 0) break;
        //对高频词进行随机下采样，丢弃掉一些高频词，能够使低频词向量更加准确，同时加快训练速度
        // 可以看作是一种平滑方法
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          //以1-ran的概率舍弃高频词
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        //如果句子长度超出最大长度则截断
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      //定位到句子头
      sentence_position = 0;
    }

    //如果当前线程处理的词数超过了它应该处理的最大值，那么开始新一轮迭代
    //如果迭代数超过上限，则停止迭代
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }

    
    word = sen[sentence_position];        // sen:当前从文件中读取的待处理的句子，存放的是每个词在词表中的索引
                                          // word: 当前词在句子中的索引
    if (word == -1) continue;
    
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;     // 初始化输入词向量为0，neu1:输入词向量，Context的向量和
    
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;    // 初始化累计误差项为0，neu1e: 输入词的误差，即 error|pred - real|
    
    
    next_random = next_random * (unsigned long long)25214903917 + 11;   // window 是一个常数 window = 5
    b = next_random % window;                         //生成一个[0, window-1]的随机数，用来确定｜context(w)｜窗口的实际宽度


    
    if (cbow) {                 // 如果使用的是CBOW模型：输入是某单词周围窗口单词的词向量，来预测该中心单词本身

      cw = 0;                   // cw:窗口长度（中心词除外）

      // 一个词的窗口为[sentence_position - window + b，sentence_position + window - b]
      // 因此窗口总长度为 2*window - 2*b + 1，单扇窗的长度是 window - b
      for (a = b; a < window * 2 + 1 - b; a++)   // 从 b 的位置开始，遍历窗口中的每个位置

        if (a != window) {                       // 去除窗口的中心词，这是我们要预测的内容，仅仅提取上下文
          c = sentence_position - window + a;    // c 代表的是正在处理的 word 在 sentence 中文位置
          if (c < 0) continue;                   // 如果当前词的位置小于0就不计算
          if (c >= sentence_length) continue;    // 如果当前词的位置大于每句话的长度也不计算
          
          last_word = sen[c];                    //sen 数组中存放的是句子中的每个词在词表中的索引
          if (last_word == -1) continue;
          
          
          for (c = 0; c < layer1_size; c++)     // 这里的 c 值得是一个词向量的某个维度，应该换一个变量名
            neu1[c] += syn0[c + last_word * layer1_size];
          //统计实际窗口中的有效词数
          cw++;
        }
      if (cw) {
        //求平均向量和
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;

        //如果采用分层softmax优化
        //根据Haffman树上从根节点到当前词的叶节点的路径，遍历所有经过的中间节点
        if (hs) {
          for (d = 0; d < vocab[word].codelen; d++) {     // d 代表的是一个单词的 Haffman 编码里面的每一个节点
            f = 0;
            
            l2 = vocab[word].point[d] * layer1_size;      // l2为当前遍历到的中间节点的向量在syn1中的起始位置         
            for (c = 0; c < layer1_size; c++){            // 对于 layer1_size 也就 model dimension 100，每一个维度都更新
              f += neu1[c] * syn1[c + l2];                // f为输入向量neu1与中间结点向量syn1的内积
            }
            if (f <= -MAX_EXP) continue;                  // 检测f有没有超出sigmoid函数表的范围
            else if (f >= MAX_EXP) continue;              // 如果没有超出范围则对f进行sigmoid变换
            
            else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];   // 通过查 expTable 表得到梯度的值
            
                                                          // f 是通过查表得到的梯度
                                                          // 注意 word2vec 中将 Haffman 编码为1的节点定义为负类，而将编码为0的节点定义为正类
                                                          // word 是词表中的每一个单词，code[d] 代表的是 word 路径上的节点
            g = (1 - vocab[word].code[d] - f) * alpha;    // g是梯度和学习率的乘积 g = (label-f)*alpha是一个负值，作用在中间节点向量上时，刚好起到调小效果
            
            for (c = 0; c < layer1_size; c++){            // 根据计算得到的修正量g和中间节点的向量更新累计误差
              neu1e[c] += g * syn1[c + l2];               // neu1e 是累积的输入词向量的误差项
            }
            // 根据计算得到的修正量g和输入向量更新中间节点的向量值
            // 很好理解，假设vocab[word].code[d]编码为1，即负类，其节点label为1-1=0
            // sigmoid函数得到的值为(0,1)范围内的数，大于label，很自然的，我们需要把这个中间节点的向量调小
            // 而此时的
            // 调小的幅度与sigmoid函数的计算值偏离label的幅度成正比
            for (c = 0; c < layer1_size; c++){
              syn1[c + l2] += g * neu1[c];                // syn1 是累积的输入词向量的误差项
            }
          }
        }
        // 如果采用负采样优化
        // 遍历所有正负样本（1个正样本 + negative个负样本）
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
            if (d == 0) {            // 第一次循环处理的是目标单词，即正样本
              target = word;
              label = 1;
            } else {                 // 除了第一次的循环都是负采样

              //从能量表中随机抽取负样本，下一个随机数就是上一个随机数乘一个大数，加上11 然后位移16个二进制位，然后再整除
              next_random = next_random * (unsigned long long)25214903917 + 11;
              target = table[(next_random >> 16) % table_size];
              if (target == 0) target = next_random % (vocab_size - 1) + 1;
              if (target == word) continue;     // 负采样取到自己了就 continue，取下一个负样本
              label = 0;
            }

            //在负采样优化中，每个词在syn1neg数组中对应一个辅助向量
            //此时的l2为syn1neg中目标单词向量的起始位置
            l2 = target * layer1_size;
            f = 0;
            //f为输入向量neu1与辅助向量的内积
            for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
            if (f > MAX_EXP) g = (label - 1) * alpha;
            else if (f < -MAX_EXP) g = (label - 0) * alpha;
            //g=(label-f)*alpha
            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            //用辅助向量和g更新累计误差
            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
            //用输入向量和g更新辅助向量
            for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
          }
        // 根据获得的累计误差，更新context(w)中每个词的词向量
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
            c = sentence_position - window + a;
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            last_word = sen[c];
            if (last_word == -1) continue;
            for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
          }
      }
    }


    //如果使用的是skip-gram模型：输入中心词，来预测该单词的上下文
    else {

      //因为需要预测context(w)中的每个词，因此需要循环2window-2b+1次遍历整个窗口
      //遍历时跳过中心词
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          //last_word为当前待预测的上下文单词
          last_word = sen[c];
          if (last_word == -1) continue;
          //l1为当前单词的词向量在syn0中的起始位置
          l1 = last_word * layer1_size;
          //初始化累计误差
          for (c = 0; c < layer1_size; c++) neu1e[c] = 0;


          // 如果采用分层softmax优化
          // 根据Haffman树上从根节点到当前词的叶节点的路径，遍历所有经过的中间节点
          if (hs) for (d = 0; d < vocab[word].codelen; d++) {
              f = 0;
              l2 = vocab[word].point[d] * layer1_size;
              // 注意，此处用到了模型对称：p(u|w)=p(w|u)，其中w为中心词，u为context(w)中每个词
              // 也就是skip-gram虽然是给中心词预测上下文，真正训练的时候还是用上下文预测中心词
              // 与CBOW不同的是这里的u是单个词的词向量，而不是窗口向量之和
              // 算法流程基本和CBOW的hs一样
              for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
              if (f <= -MAX_EXP) continue;
              else if (f >= MAX_EXP) continue;
              else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
              // 'g' is the gradient multiplied by the learning rate
              g = (1 - vocab[word].code[d] - f) * alpha;
              // Propagate errors output -> hidden
              for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
              // Learn weights hidden -> output
              for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
            }
          // 如果采用负采样优化
          // 遍历所有正负样本（1个正样本+negative个负样本）
          // 算法流程基本和CBOW的ns一样，也采用的是模型对称
          if (negative > 0) for (d = 0; d < negative + 1; d++) {
              if (d == 0) {
                target = word;
                label = 1;
              } else {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                target = table[(next_random >> 16) % table_size];
                if (target == 0) target = next_random % (vocab_size - 1) + 1;
                if (target == word) continue;
                label = 0;
              }
              l2 = target * layer1_size;
              f = 0;
              for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
              if (f > MAX_EXP) g = (label - 1) * alpha;
              else if (f < -MAX_EXP) g = (label - 0) * alpha;
              else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
              for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
              for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
            }
          // Learn weights input -> hidden
          for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        }
    }
    //完成一个词的训练，句子中位置往后移一个词
    sentence_position++;
    //处理完一个句子后，将句子长度置为0，进入循环，重新读区句子并进行逐词计算
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

//完整的模型训练流程函数

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  //创建多线程，线程数为num_threads
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  //设置初始学习率
  starting_alpha = alpha;
  //如果有词汇表文件，则从中加载生成词表和hash表，否则从训练文件中获得
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  //根据需要，可以将词表中的词和词频输出到文件
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  //初始化训练网络
  InitNet();
  //如果采用负采样优化，则需要初始化能量表
  if (negative > 0) InitUnigramTable();
  //开始计时
  start = clock();
  //创建训练线程
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  //如果classes参数为0，则输出所有词向量到文件中
  if (classes == 0) {
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  }
  //如果classes参数不为0，则需要对词向量进行K-means聚类，输出词类
  //classes为最后要分成的类的个数

  else {
    //clcn：总类数
    //iter：总迭代次数
    //closeid：用来存储计算过程中离某个词最近的类编号
    // 在词向量上进行K—means聚类
    int clcn = classes, iter = 10, closeid;
    //centcn：属于每个类的单词数
    int *centcn = (int *)malloc(classes * sizeof(int));
    //cl：每个单词所属的类编号
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    //x:用来存储每次计算得到的词向量和类中心的内积，值越大说明距离越近
    //closev：最大的内积，即距离最近
    real closev, x;
    //cent：每个类的中心向量
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    //先给所有单词随机指派类
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;

    //一共迭代iter次
    for (a = 0; a < iter; a++) {
      //初始化类中心向量数组为0
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      //初始化每个类含有的单词数为1
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      //将刚才随意分配的所属于同一个类的词向量相加，并且计算属于每个类的词数
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          //计算每个类的平均中心向量
          cent[layer1_size * b + c] /= centcn[b];
          //closev为类平均中心向量的二范数的平方
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        //对closev开方，此时的closev即为类平均中心向量的二范数
        closev = sqrt(closev);
        //用得到的范数对中心向量进行归一化
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }

      //遍历词表中的每个词，为其中心分配距离最近的类
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          //对词向量和归一化的类中心向量做内积
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          //内积越大说明两点之间距离越近
          //取所有类中与这个词的词向量内积最大的一个类，将词分到这个类中
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }

    // 经过多次迭代后，逐渐会将词向量向正确的类靠拢
    // 输出K-means聚类结果到文件中
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}
//当参数缺失时，输出提示信息
int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
