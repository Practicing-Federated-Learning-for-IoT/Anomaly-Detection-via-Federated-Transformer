# Anomaly Detection via Federated Transformer
## Dataset
We conduct the expermients in four datasets, NSL-KDD, Spambase, Shuttle, Arrhythmia. The details of datasets as follows:
![](https://codimd.xixiaoyao.cn/uploads/upload_80e50175c49182f9935ce969de97356e.png)
Consider the real world settings, we set the number of labeled anomalies is 30 in NSL-KDD, Spambase and Shuttle, we set the number of labeled anomalies is 15 in Arrhythmia. 
## Framework
We propose the framework as follows:
![](https://codimd.xixiaoyao.cn/uploads/upload_45e7974b68f16c42a1e92e8702a48260.png)
## How to run
We want to run the experiment on NSL-KDD via the followinng commend lines: 
```python=
python main.py -d nslkdd_normalization.csv -c 3 -f 1 -d_data 122 -heads 2 -r 0.0005 -e 1000 -d_feature 64
```
-d means the dataset which you want to use
-c means the number of clients
-f means the portion of clients
-d_data means the dimension of original data
-d_feature means the dimension of feature uploaded by clients
-heads means the number of mulit-head attention blocks
-r means the percentage of labeled anomalies in the training datasets
-e means the number of epoch in framework
