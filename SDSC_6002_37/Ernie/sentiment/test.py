import pandas as  pd
import numpy as np
import paddlenlp
import paddle
import os
import random
from paddlenlp.datasets import MapDataset
from paddlenlp.datasets import load_dataset
import functools
import numpy as np
from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_function(examples, tokenizer, max_seq_length, is_test=False):

    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    if not is_test:
        result["labels"] = examples["label"]
    return result
def predict(model,test_data_loader):
    result = []
    model.eval()
    for batch in test_data_loader:
            input_ids, token_type_ids = batch['input_ids'], batch['token_type_ids']
            logits = model(input_ids, token_type_ids)
            y_pre = logits.numpy().argmax(axis=-1)
            result.extend(y_pre)
    return np.array(result)

def read(data_path):
    df = pd.read_csv(data_path)
    for i in range(len(df)):
        yield {'text': df.iloc[i, 2], 'label': df.iloc[i, 1]}

model_name = "ernie-3.0-base-zh"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=512)
collate_fn = DataCollatorWithPadding(tokenizer)
stata_param = paddle.load('/home/Courses/sdsc6001/sdsc6001_008/6002/Ernie/sentiment/ernie_ckpt_base/model_state.pdparams')
model.set_state_dict(stata_param)
test_ds = load_dataset(read, data_path='/home/Courses/sdsc6001/sdsc6001_008/6002/Data/sentiment/emotion_test_data.csv',lazy=False)
test_ds = test_ds.map(trans_func)
test_batch_sampler = BatchSampler(test_ds, batch_size=32, shuffle=False)
test_data_loader = DataLoader(dataset=test_ds, batch_sampler=test_batch_sampler, collate_fn=collate_fn)
y_pre = predict(model,test_data_loader)

df=pd.read_csv('/home/Courses/sdsc6001/sdsc6001_008/6002/Data/sentiment/emotion_test_data.csv')
y_true=np.array(df['label'])
print(classification_report(y_true,y_pre))
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(y_true,y_pre),annot=True)
plt.show()