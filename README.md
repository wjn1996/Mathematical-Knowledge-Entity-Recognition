# Mathematical-Knowledge-Entity-Recognition
<b>1.Introduction</b>

This is a novel project for mathematical knowledge entity recognition. The algorithm is mainly model by BiLSTM+CRF with Chinese Word Embeddings. This project is the first process for Mathematical Knowledge Graph(Math-KG).

<b>2.Copyright Notice</b>

Leader: WangJianing<br>
Email:lygwjn@126.com<br>
CSDN:https://blog.csdn.net/qq_36426650<br>

<b>3.Project Overview</b>
This code contains 6 files, such as:<br>
main.py——which you can run by python;<br>
model.py——which contains our models;<br>
eval.py——which contains the procedure to run perl code;<br>
data.py——mainly preprocess dataset;<br>
run_mainpy——mainly generate pkl file about all vocabs;<br>
utils.py——contains the decoder on labels.<br>

<b>4.Useage Details</b>
After download this project, you had better follow these steps to run our program.

(1)Firstly, you should download the mathematical NLP datasets:https://blog.csdn.net/qq_36426650/article/details/87719204, This webpage also has very detail notes about how to use this dataset. This page is in Chinese, if you don't know Chinese, you can have your browser translate it into the language you want to view the entire blog information.

The datasets contains two subject:Junior middle school mathematics & high middle school mathematics. Each dataset consists of two files, the training set "ner_train_data" and the test set "ner_test_data". 

You should alse download the Chines word embeddings. We have pretrained word embeddings from Wikipedia such as word2vec, glove and gwe with 300-dimension.

(2)Secondly, you should create a new directionary to store the dataset.

(3)Please open file run_main.py and edit the variable "file" value. And then generate a word2id.pkl file by run:
```
python3 run_main.py "<dataset dictionary>"
```
for example, if you create a fold named "math" and put datasets in it, the command is "python3 run_main.py math"

After that, you will achieve a new file named "word2id.pkl".

(4)If you want to train by yourslef, you can run:
```
python3 main.py --train_data=<trainset file name> --test=<testset file name> --CRF=<True or False> --embedding_type=<word embedding type> --mode="train"
```

