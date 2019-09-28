import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding, load_embeddings


## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.333  # need ~700MB GPU memory
file = 'math_data'

## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default=file, help='train data source')
parser.add_argument('--test_data', type=str, default=file, help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=20, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--embedding_type', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
args = parser.parse_args()


## get char embeddings
#word2id:为每一个不重复的字进行编号，其中UNK为最后一位
word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
print("\n========word2id=========\n",word2id)
if args.embedding_type == 'random':
    #随机生成词嵌入矩阵（一共3905个字，默认取300个特征，维度为3905*300）
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embeddings = load_embeddings(args.embedding_dim,word2id,args.embedding_type)
    #使用gensim（word2vec）基于wiki百科语料训练的中文词向量
    
print("\n=========embeddings==========\n",embeddings,"\ndim(embeddings)=",embeddings.shape)

## read corpus and get training data获取
if args.mode != 'demo':
    train_path = os.path.join('.', args.train_data, 'ner_train_data')
    test_path = os.path.join('.', args.test_data, 'ner_test_data')
    train_data = read_corpus(train_path)#读取训练集
    test_data = read_corpus(test_path); test_size = len(test_data)#读取测试集
    print('train_data=\n',train_data)
    #print("\n==========train_data================\n",train_data)
    #print("\n==========test_data================\n",test_data)


## paths setting创建相应文件夹目录

paths = {}
# 时间戳就是一个时间点，一般就是为了在同步更新的情况下提高效率之用。
#就比如一个文件，如果他没有被更改，那么他的时间戳就不会改变，那么就没有必要写回，以提高效率，
#如果不论有没有被更改都重新写回的话，很显然效率会有所下降。
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
#输出路径output_path路径设置为data_path_save下的具体时间名字为文件名
output_path = os.path.join('.', args.train_data+"_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)

summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


## training model
if args.mode == 'train':
    #创建对象model
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    #创建结点，
    model.build_graph()

    ## hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    ## train model on the whole training data
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print('ckpt_file=',ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while(1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())#例：['在', '弄', '恩', '哦', '呜']
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]#例：[(['在', '弄', '恩', '哦', '呜'], ['o', 'o', 'o', 'o', 'o'])]
                tag = model.demo_one(sess, demo_data)
                PER, LOC, ORG = get_entity(tag, demo_sent)
                print('知识点实体: {}'.format(PER),'\n定律法则实体: {}'.format(LOC))
