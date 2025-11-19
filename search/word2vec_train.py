import jieba
from gensim.models import Word2Vec
import PDF_reader
# file_path='./default.pdf'
with open("extracted_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

#分词
sentences = []
for line in text.split('。'):
    if line.strip():
        words = list(jieba.cut(line.strip()))
        sentences.append(words)

print(f"分词完成，共 {len(sentences)} 个句子")

#训练Word2Vec模型
model = Word2Vec(
    sentences=sentences,
    vector_size=600,    # 向量维度
    window=10,           # 上下文窗口
    min_count=2,        # 最小词频
    workers=4,          # 线程数
    sg=0               # 1=skip-gram, 0=CBOW
)

print("Word2Vec训练完成!")

#保存模型
model.save("word2vec.model")
print("模型已保存为: word2vec.model")