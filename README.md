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

<b>4.Notes</b>

We define two kinds entities such as "KNOW" and "PRIN". "KNOW" represents the objective mathematical knowledge, while "PRIN" denotes the abstract mathematical theorem or method. We use "B" represents the first character in an entity, and "I" represents other characters. Non-entities are marked as "O".

<b>5.Useage Details</b>
After download this project, you had better follow these steps to run our program.

(1) Firstly, you should download the mathematical NLP datasets:https://blog.csdn.net/qq_36426650/article/details/87719204, This webpage also has very detail notes about how to use this dataset. This page is in Chinese, if you don't know Chinese, you can have your browser translate it into the language you want to view the entire blog information.

The datasets contains two subject:Junior middle school mathematics & high middle school mathematics. Each dataset consists of two files, the training set "ner_train_data" and the test set "ner_test_data". 

You should alse download the Chines word embeddings. We have pretrained word embeddings from Wikipedia such as word2vec, glove and gwe with 300-dimension.

(2) Secondly, you should create a new directionary to store the dataset.

(3) Please open file run_main.py and edit the variable "file" value. And then generate a word2id.pkl file by run:
```
python3 run_main.py "<dataset dictionary>"
```
for example, if you create a fold named "math" and put datasets in it, the command is "python3 run_main.py math"

After that, you will achieve a new file named "word2id.pkl".

(4) If you want to train by yourslef, you can run:
```
python3 main.py --train_data=<trainset file name> --CRF=<True or False> --embedding_type=<word embedding type> --mode="train"
```
For example:
```
python3 main.py --train_data="highmath_data" --CRF=True --embedding_type="glove" --mode="train"
```
The model will be stored in new files.

Of course, you can change some hyper-parameters in file main.py.

(5) After training processing, you can run test.py to evalue the model by PRF1.
```
python3 main.py --test=<testset file name> --CRF=<True or False> --embedding_type=<word embedding type> --mode="test"
```
for example:
```
python3 main.py --test_data="highmath_data" --CRF=True --embedding_type="glove" --mode="test"
```

(6) We also provide a demo that you can feed only one sentence to the model. you can run:
```
python3 main.py --mode="demo" --demo_model=<model dictionary>
```
You can input a sentence into command and then the model can return all entitys with tags.

<b>5.Result Demonstration</b>
The result of our pretrain model is shown as follow:
![Alt text](https://github.com/wjn1996/Mathematical-Knowledge-Entity-Recognition/blob/master/images/evaluation.png)
The demonstration of our demo is shown:
![Alt text](https://github.com/wjn1996/Mathematical-Knowledge-Entity-Recognition/blob/master/images/demo.png)
 
