import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 读取训练数据和测试数据
train_df = pd.read_csv('/home/Courses/sdsc6001/sdsc6001_008/6002/Data/sentiment/emotion_train_data.csv')
test_df = pd.read_csv('/home/Courses/sdsc6001/sdsc6001_008/6002/Data/sentiment/emotion_test_data.csv')

# 数据清洗
def clean_text(text):
    # 去除HTML标签和特殊字符
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]+', '', text)
    # 分词和去停用词
    words = [word for word in jieba.cut(text) if word.strip() and word not in stop_words]
    # 过滤文本长度
    if len(words) < min_len or len(words) > max_len:
        return None
    return ' '.join(words)

# 定义停用词和文本长度阈值
stop_words = set(open('stopwords.txt', encoding='utf-8').read().strip().split('\n'))
min_len = 5
max_len = 200

# 对训练数据和测试数据进行数据清洗
train_df['text_a'] = train_df['text_a'].apply(clean_text)
test_df['text_a'] = test_df['text_a'].apply(clean_text)
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# 对文本进行向量化
vectorizer = CountVectorizer()
train_X = vectorizer.fit_transform(train_df['text_a'])
test_X = vectorizer.transform(test_df['text_a'])

# 获取标签
train_y = train_df['label']
test_y = test_df['label']

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_X, train_y)

# 在测试数据上进行预测
pred_y = rf.predict(test_X)

# 计算准确率和分类报告
acc = accuracy_score(test_y, pred_y)
report = classification_report(test_y, pred_y)

print(f"Accuracy: {acc}")
print(f"Classification Report:\n{report}")