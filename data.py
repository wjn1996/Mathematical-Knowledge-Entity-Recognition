#数据处理

import sys, pickle, os, random
import numpy as np
import gensim #add by wjn

## tags, BIO
tag2label = {"O": 0,
             "B-KNOW": 1, "I-KNOW": 2,
             "B-PRIN": 3, "I-PRIN": 4,
             "B-OTHER": 5, "I-OTHER": 6
             }

#输入train_data文件的路径，读取训练集的语料，输出train_data
def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            # [char, label] = line.split(' ')
            [char, label] = line.replace('\n','').split(' ')
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data

#生成word2id序列化文件
def vocab_build(vocab_path, corpus_path, min_count):
    """
	#建立词汇表
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    #读取数据（训练集或测试集）
    #data格式：[(字,标签),...]
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    #将任意对象进行序列化保存
    print('word2id:\n',word2id)
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)

#将句子中每一个字转换为id编号，例如['我','爱','中','国'] ==> ['453','7','3204','550']
def sentence2id(sent, word2id):
    """

    :param sent:源句子
    :param word2id:对应的转换表
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

#读取word2id文件
def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    #反序列化
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id

#随机嵌入
def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:随机对列表data进行排序
    :return:
    """
    #如果参数shuffle为true，则对data列表进行随机排序
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
    	#将句子转换为编号组成的数字序列
        sent_ = sentence2id(sent_, vocab)
        #将标签序列转换为数字序列
        label_ = [tag2label[tag] for tag in tag_]
        #一个句子就是一个样本，当句子数量等于预设的一批训练集数量，便输出该样本
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

#add by 王嘉宁
#加载训练好的模型词向量,包括三类词向量w2v，glove，GWE
#add by wjn
def load_embeddings(embedding_dim, vocab, embedding_type):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab), embedding_dim).astype(np.float32) / np.sqrt(len(vocab))
    # load any vectors from the word2vec
    # print("Load glove file {0}".format(embedding_path))
    embedding_dir = './embeddings/'
    if embedding_type == "glove":
        f = open(embedding_dir + 'wiki.zh.glove.Mode', 'r', encoding='utf8')
        for line in f:
            splitLine = line.split(' ')
            word = splitLine[0]
            embedding = np.asarray(splitLine[1:], dtype='float32')
            if word in vocab:
                idx = vocab[word]
                if idx != 0:
                    initW[idx] = embedding
    elif embedding_type == "word2vec":
        model = gensim.models.Word2Vec.load(embedding_dir + 'wiki.zh.w2v.Mode')
        allwords = model.wv.vocab
        for word in allwords:
            embedding = np.asarray(model[word], dtype='float32')
            if word in vocab:
                idx = vocab[word]
                if idx != 0:
                    initW[idx] = embedding
    elif embedding_type == "gwe":
        with open(embedding_dir + 'wiki.zh.GWE.mode','r',encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                splitLine = line.split(' ')
                word = splitLine[0]
                embedding = np.asarray(splitLine[1:301], dtype='float32')
                if word in vocab:
                    idx = vocab[word]
                    if idx != 0:
                        initW[idx] = embedding
    return initW