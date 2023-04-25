import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
#加载数据：
# 从csv文件中读取数据
train_df = pd.read_csv("/home/Courses/sdsc6001/sdsc6001_008/6002/Data/product/product_train_data.csv")
dev_df = pd.read_csv("/home/Courses/sdsc6001/sdsc6001_008/6002/Data/product/product_dev_data.csv")
test_df = pd.read_csv("/home/Courses/sdsc6001/sdsc6001_008/6002/Data/product/product_test_data.csv")

# 分离出标签和文本
train_labels = train_df["label"].values
train_texts = train_df["text_a"].values

dev_labels = dev_df["label"].values
dev_texts = dev_df["text_a"].values

test_labels = test_df["label"].values
test_texts = test_df["text_a"].values

# 将标签转换为0和1
train_labels = train_labels.astype(int)
dev_labels = dev_labels.astype(int)
test_labels = test_labels.astype(int)
#将文本数据转换为BERT输入格式：
# 加载BERT tokenizer，用于将文本转换为tokens
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True,max_length=512)

max_length = 128

# 将文本转换为tokens
train_tokenized_texts = [tokenizer.tokenize(text)[:max_length-2] for text in train_texts]
dev_tokenized_texts = [tokenizer.tokenize(text)[:max_length-2] for text in dev_texts]
test_tokenized_texts = [tokenizer.tokenize(text)[:max_length-2] for text in test_texts]

# 将tokens转换为id
train_input_ids = [tokenizer.encode_plus(txt, add_special_tokens=True, max_length=max_length, truncation=False, pad_to_max_length=True)['input_ids'] for txt in train_tokenized_texts]
dev_input_ids = [tokenizer.encode_plus(txt, add_special_tokens=True, max_length=max_length, truncation=False, pad_to_max_length=True)['input_ids'] for txt in dev_tokenized_texts]
test_input_ids = [tokenizer.encode_plus(txt, add_special_tokens=True, max_length=max_length, truncation=False, pad_to_max_length=True)['input_ids'] for txt in test_tokenized_texts]

# 将输入转换为tensor
train_input_ids = torch.tensor(train_input_ids)
dev_input_ids = torch.tensor(dev_input_ids)
test_input_ids = torch.tensor(test_input_ids)

# 对输入进行填充
train_attention_masks = torch.tensor([[1 if token > 0 else 0 for token in txt] for txt in train_input_ids])
dev_attention_masks = torch.tensor([[1 if token > 0 else 0 for token in txt] for txt in dev_input_ids])
test_attention_masks = torch.tensor([[1 if token > 0 else 0 for token in txt] for txt in test_input_ids])

# 将标签转换为tensor
train_labels = torch.tensor(train_labels)
dev_labels = torch.tensor(dev_labels)
test_labels = torch.tensor(test_labels)

# 创建数据集
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
#定义模型：
# 加载预训练的BERT模型，同时设置为分类模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 将模型设置为训练模式
model.train()
#定义训练参数：
# 定义训练参数
batch_size = 16
epochs = 10
learning_rate = 1e-5

# 定义优化器
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#训练模型：
#训练模型：
best_dev_f1 = 0.0
best_model = None

# 训练模型
for epoch in range(epochs):
    for batch in train_dataloader:
        # 取出数据
        batch_input_ids, batch_attention_masks, batch_labels = batch

        # 将数据传入模型中进行训练
        outputs = model(batch_input_ids, 
                        token_type_ids=None, 
                        attention_mask=batch_attention_masks, 
                        labels=batch_labels)
        loss = outputs.loss
        logits = outputs.logits

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 清空梯度
        optimizer.zero_grad()

    # 在开发集上评估模型
    model.eval()
    with torch.no_grad():
        total_dev_loss = 0
        total_dev_accuracy = 0
        total_dev_f1 = 0.0
        for batch in dev_dataloader:
            # 取出数据
            batch_input_ids, batch_attention_masks, batch_labels = batch

            # 将数据传入模型中进行评估
            outputs = model(batch_input_ids, 
                            token_type_ids=None, 
                            attention_mask=batch_attention_masks, 
                            labels=batch_labels)
            loss = outputs.loss
            logits = outputs.logits

            # 累积损失、准确率和F1值
            total_dev_loss += loss.item()
            total_dev_accuracy += (logits.argmax(1) == batch_labels).sum().item()
            total_dev_f1 += f1_score(batch_labels.cpu(), logits.argmax(1).cpu(), average='macro')

        # 计算平均损失、准确率和F1值
        avg_dev_loss = total_dev_loss / len(dev_dataloader)
        avg_dev_accuracy = total_dev_accuracy / len(dev_dataset)
        avg_dev_f1 = total_dev_f1 / len(dev_dataloader)

        # 打印并记录最佳模型
        print(f"Epoch {epoch + 1}:")
        print(f"  Train loss: {loss.item():.4f}")
        print(f"  Dev loss: {avg_dev_loss:.4f}")
        print(f"  Dev accuracy: {avg_dev_accuracy:.4f}")
        print(f"  Dev F1: {avg_dev_f1:.4f}")
        if avg_dev_f1 > best_dev_f1:
            best_dev_f1 = avg_dev_f1
            best_model = model.state_dict()
            print('best model update')
if best_model is not None:
    # 将模型权重保存到文件中
    torch.save(best_model, 'best_model.pt')
    print('best model is saved')
else:
    print("No best model found.")
# 加载最佳模型并在测试集上进行测试
model.load_state_dict(best_model)
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for batch in test_dataloader:
        batch_input_ids, batch_attention_masks, batch_labels = batch
        outputs = model(batch_input_ids, 
                        token_type_ids=None, 
                        attention_mask=batch_attention_masks, 
                        labels=batch_labels)
        logits = outputs.logits
        y_pred += logits.argmax(1).tolist()
        y_true += batch_labels.tolist()

# 输出 classification_report
print(classification_report(y_true, y_pred))