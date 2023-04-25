import pandas as  pd
import numpy as np
import paddlenlp
import paddle
#import os
import random
from paddlenlp.datasets import MapDataset
from paddlenlp.datasets import load_dataset
import functools
import numpy as np
from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

import pandas as pd


def evaluate(model, loss_func, metric_func, valid_data_loader):
    loss_1 = []
    metric_func.reset()
    model.eval()
    for step, batch in enumerate(valid_data_loader, start=1):
        input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']
        # 计算模型输出、损失函数值、分类概率值、准确率
        logits = model(input_ids, token_type_ids)
        loss = loss_func(logits, labels)
        loss_1.append(loss.numpy()[0])
        probs = F.softmax(logits, axis=1)
        correct = metric_func.compute(probs, labels)
        metric.update(correct)
        acc = metric_func.accumulate()
    loss_eval = np.mean(loss_1)
    acc_eval = acc
    print(f'----------------------loss:{loss_eval},acc:{acc_eval}----------------------')
    metric_func.reset()
    model.train()
    return acc_eval

def read(data_path):
            df=pd.read_csv(data_path)
            for i in range(len(df)):
                yield {'text': df.iloc[i,2], 'label': df.iloc[i,1]}


# 数据预处理函数，利用分词器将文本转化为整数序列
def preprocess_function(examples, tokenizer, max_seq_length, is_test=False):

    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    if not is_test:
        result["labels"] = examples["label"]
    return result

# data_path为read()方法的参数
train_ds = load_dataset(read, data_path='/home/Courses/sdsc6001/sdsc6001_008/6002/Data/product/product_train_data.csv',lazy=False)
#test_ds = load_dataset(read, data_path='/home/Courses/sdsc6001/sdsc6001_008/6002/test_data.csv',lazy=False)
dev_ds = load_dataset(read, data_path='/home/Courses/sdsc6001/sdsc6001_008/6002/Data/product/product_dev_data.csv',lazy=False)


model_name = "ernie-3.0-base-zh"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = paddle.optimizer.AdamW(learning_rate=4e-5, parameters=model.parameters())
criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()




trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=512)
train_ds = train_ds.map(trans_func)
dev_ds = dev_ds.map(trans_func)

# collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠
collate_fn = DataCollatorWithPadding(tokenizer)

# 定义BatchSampler，选择批大小和是否随机乱序，进行DataLoader
train_batch_sampler = BatchSampler(train_ds, batch_size=64, shuffle=True)
dev_batch_sampler = BatchSampler(dev_ds, batch_size=64, shuffle=False)
train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
dev_data_loader = DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=collate_fn)


import time
import paddle.nn.functional as F
import os

epochs = 10 # 训练轮次
ckpt_dir = "ernie_ckpt_base_product" #训练过程中保存模型参数的文件夹
best_acc = 0
best_step = 0
global_step = 0 #迭代次数
tic_train = time.time()

for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']

        # 计算模型输出、损失函数值、分类概率值、准确率
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        # 每迭代10次，打印损失函数值、准确率、计算速度
        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                   10 / (time.time() - tic_train)))
            tic_train = time.time()

        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

    save_dir = ckpt_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(global_step, end=' ')

    acc_eval = evaluate(model, criterion, metric, dev_data_loader)

    if acc_eval > best_acc:
        best_acc = acc_eval
        best_step = global_step
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
