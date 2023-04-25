import pandas as pd
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import accuracy_score, classification_report

# 读取训练数据、验证数据和测试数据
train_df = pd.read_csv('/home/Courses/sdsc6001/sdsc6001_008/6002/Data/product/product_train_data.csv')
dev_df = pd.read_csv('/home/Courses/sdsc6001/sdsc6001_008/6002/Data/product/product_dev_data.csv')
test_df = pd.read_csv('/home/Courses/sdsc6001/sdsc6001_008/6002/Data/product/product_test_data.csv')
#数据清洗
# 对文本进行分词和向量化
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['text_a'])
train_X = tokenizer.texts_to_sequences(train_df['text_a'])
dev_X = tokenizer.texts_to_sequences(dev_df['text_a'])
test_X = tokenizer.texts_to_sequences(test_df['text_a'])

# 填充序列
max_len = 100
train_X = pad_sequences(train_X, maxlen=max_len, padding='post')
dev_X = pad_sequences(dev_X, maxlen=max_len, padding='post')
test_X = pad_sequences(test_X, maxlen=max_len, padding='post')

# 获取标签
train_y = train_df['label']
dev_y = dev_df['label']
test_y = test_df['label']

# 构建模型
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128, input_length=max_len))
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_X, train_y, epochs=10, batch_size=32, validation_data=(dev_X, dev_y))

# 在测试数据上进行预测
pred_y = (model.predict(test_X) > 0.5).astype(int)

# 计算准确率和分类报告
acc = accuracy_score(test_y, pred_y)
report = classification_report(test_y, pred_y)

print(f"Accuracy: {acc}")
print(f"Classification Report:\n{report}")
