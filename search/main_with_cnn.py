import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
import jieba

# 1. CNN特征提取器
class CNNFeatureExtractor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters=100, filter_sizes=[2, 3, 4]):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        
        # 卷积层：不同大小的卷积核捕获不同长度的短语特征
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs, padding=1)
            for fs in filter_sizes
        ])
        
        # 全连接层
        self.fc = nn.Linear(num_filters * len(filter_sizes), embedding_dim)
        
    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        
        # 应用不同大小的卷积核
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # [batch_size, num_filters, seq_len]
            # 全局最大池化
            pooled = F.adaptive_max_pool1d(conv_out, 1)  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [batch_size, num_filters]
        
        # 合并所有卷积特征
        combined = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        
        # 投影回原始维度
        output = self.fc(combined)  # [batch_size, embedding_dim]
        return F.normalize(output, p=2, dim=1)  # L2归一化

# 2. 语义搜索引擎
class CNNSemanticSearch:
    def __init__(self, word2vec_model):
        self.model = word2vec_model
        self.vector_size = word2vec_model.vector_size
        self.cnn_extractor = CNNFeatureExtractor(
            vocab_size=len(word2vec_model.wv),
            embedding_dim=self.vector_size
        )
        
    def text_to_cnn_vector(self, text, max_length=50):
        """使用CNN提取文本特征向量"""
        words = list(jieba.cut(text))[:max_length]  # 限制长度
        
        # 创建词向量序列
        word_vectors = []
        for word in words:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])
        
        if not word_vectors:
            return np.zeros(self.vector_size)
        
        # 填充或截断到固定长度
        if len(word_vectors) < max_length:
            padding = [np.zeros(self.vector_size)] * (max_length - len(word_vectors))
            word_vectors.extend(padding)
        else:
            word_vectors = word_vectors[:max_length]
        
        # 转换为tensor
        text_tensor = torch.FloatTensor(np.array(word_vectors)).unsqueeze(0)  # [1, seq_len, emb_dim]
        
        # 通过CNN提取特征
        with torch.no_grad():
            cnn_vector = self.cnn_extractor(text_tensor)
        
        return cnn_vector.squeeze(0).numpy()

# 3. 加载数据并构建搜索引擎
print("正在构建CNN搜索引擎...")

# 加载Word2Vec模型
model = Word2Vec.load("word2vec.model")
print(f"Word2Vec模型加载完成，词汇表大小: {len(model.wv)}")

# 加载文档
with open("extracted_text.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

print(f"加载了 {len(documents)} 个文档")

# 创建搜索引擎实例
search_engine = CNNSemanticSearch(model)

# 预计算所有文档的CNN向量
print("正在计算文档CNN向量...")
doc_vectors = []
for i, doc in enumerate(documents):
    if i % 100 == 0:
        print(f"已处理 {i}/{len(documents)} 个文档")
    vec = search_engine.text_to_cnn_vector(doc)
    doc_vectors.append(vec)

doc_vectors = np.array(doc_vectors)
print("文档向量计算完成!")

# 4. 搜索函数
def cnn_search(query, top_k=5):
    """使用CNN特征进行搜索"""
    # 提取查询的CNN向量
    query_vector = search_engine.text_to_cnn_vector(query)
    
    # 计算余弦相似度
    similarities = np.dot(doc_vectors, query_vector)
    
    # 获取最相似的文档
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append({
                'score': float(similarities[idx]),
                'text': documents[idx],
                'doc_id': int(idx)
            })
    
    return results

# 5. 对比搜索函数（使用原始Word2Vec平均向量）
def word2vec_search(query, top_k=5):
    """使用原始Word2Vec平均向量搜索"""
    words = list(jieba.cut(query))
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if vectors:
        query_vector = np.mean(vectors, axis=0)
        query_vector = query_vector / np.linalg.norm(query_vector)
    else:
        query_vector = np.zeros(model.vector_size)
    
    # 计算所有文档的平均向量
    doc_w2v_vectors = []
    for doc in documents:
        words = list(jieba.cut(doc))
        vecs = [model.wv[w] for w in words if w in model.wv]
        if vecs:
            doc_vec = np.mean(vecs, axis=0)
            doc_vec = doc_vec / np.linalg.norm(doc_vec)
        else:
            doc_vec = np.zeros(model.vector_size)
        doc_w2v_vectors.append(doc_vec)
    
    doc_w2v_vectors = np.array(doc_w2v_vectors)
    similarities = np.dot(doc_w2v_vectors, query_vector)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append({
                'score': float(similarities[idx]),
                'text': documents[idx],
                'doc_id': int(idx)
            })
    
    return results

# 6. 交互式搜索
print("\n=== CNN语义搜索引擎就绪 ===")
print("输入 'quit' 退出程序")
print("输入 'compare 搜索词' 对比两种方法")

while True:
    user_input = input("\n🔍 请输入搜索词: ").strip()
    
    if user_input.lower() == 'quit':
        print("再见!")
        break
    
    if not user_input:
        continue
    
    if user_input.startswith('compare '):
        query = user_input[8:]
        print(f"\n=== 对比搜索: '{query}' ===")
        
        print("\n🔬 CNN搜索结果:")
        cnn_results = cnn_search(query, top_k=3)
        for i, result in enumerate(cnn_results, 1):
            print(f"{i}. [CNN相似度: {result['score']:.3f}] {result['text'][:80]}...")
        
        print("\n📊 Word2Vec平均向量搜索结果:")
        w2v_results = word2vec_search(query, top_k=3)
        for i, result in enumerate(w2v_results, 1):
            print(f"{i}. [W2V相似度: {result['score']:.3f}] {result['text'][:80]}...")
    
    else:
        results = cnn_search(user_input)
        
        if results:
            print(f"\n找到 {len(results)} 个相关结果:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. 📄 [CNN相似度: {result['score']:.3f}]")
                print(f"   文档 {result['doc_id']}: {result['text'][:100]}...")
        else:
            print("没有找到相关结果")