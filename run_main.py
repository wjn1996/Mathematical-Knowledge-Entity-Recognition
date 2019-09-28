#生成pkl文件
import data
# import re
file = 'highmath_data'

data.vocab_build('./' + file + '/word2id.pkl', './' + file +'/ner_train_data', 1)
