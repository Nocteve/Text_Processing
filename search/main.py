import numpy as np
from gensim.models import Word2Vec
import jieba
import PDF_reader
# 1. 加载模型和数据
model = Word2Vec.load("word2vec.model")
with open("extracted_text.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

print(f"加载了 {len(documents)} 个文档")

# 2. 文档向量化函数
def text_to_vector(text):
    words = list(jieba.cut(text))
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if vectors:
        vector = np.mean(vectors, axis=0)
        # 归一化向量
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
    else:
        return np.zeros(model.vector_size)

# 3. 预先计算所有文档向量
doc_vectors = []
for doc in documents:
    vec = text_to_vector(doc)
    doc_vectors.append(vec)

doc_vectors = np.array(doc_vectors)
print("文档向量计算完成!")

# 4. 余弦相似度搜索
def search(query, top_k=5):
    query_vector = text_to_vector(query)
    
    # 计算余弦相似度
    similarities = np.dot(doc_vectors, query_vector)
    
    # 获取最相似的文档索引
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:  # 只返回正相似度的结果
            results.append({
                'score': float(similarities[idx]),
                'text': documents[idx],
                'doc_id': int(idx)
            })
    
    return results

# 5. 交互式搜索
print("\n=== 搜索引擎就绪 ===")
print("基于余弦相似度的语义搜索")
print("输入 'quit' 退出程序")

while True:
    query = input("\n请输入搜索词: ").strip()
    
    if query.lower() == 'quit':
        print("再见!")
        break
    
    if not query:
        continue
    
    results = search(query)
    
    if results:
        print(f"\n找到 {len(results)} 个相关结果:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 📄 [相似度: {result['score']:.3f}]")
            print(f"   文档 {result['doc_id']}: {result['text'][:120]}...")
    else:
        print("没有找到相关结果")