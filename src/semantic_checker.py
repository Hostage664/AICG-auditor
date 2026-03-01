"""
语义检测模块 - 基于FAISS向量相似度
"""

import json
import numpy as np
from typing import List, Dict, Tuple
import re
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 延迟导入，避免启动时加载
_faiss = None
_model = None


def _load_faiss():
    """延迟加载FAISS"""
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _load_model():
    """延迟加载嵌入模型"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        import os
        
        # 使用国内镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        print("正在加载嵌入模型（首次运行需要下载，约100MB）...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print("模型加载完成")
    return _model


class SemanticChecker:
    def __init__(self, db_path: str = "config/synonyms_db.json"):
        with open(db_path, 'r', encoding='utf-8') as f:
            self.db = json.load(f)
        
        self.categories = self.db['categories']
        self.dimension = self.db['metadata']['dimension']
        
        # 构建索引
        self.indices = {}      # FAISS索引
        self.entries = {}      # 原始条目
        self._build_indices()
    
    def _build_indices(self):
        """为每个类别构建FAISS向量索引"""
        model = _load_model()
        faiss = _load_faiss()
        
        for cat_name, cat_data in self.categories.items():
            entries = cat_data['entries']
            texts = [e['text'] for e in entries]
            
            # 编码为向量
            print(f"构建 [{cat_name}] 索引: {len(texts)} 条...")
            embeddings = model.encode(texts, convert_to_numpy=True)
            embeddings = embeddings.astype('float32')
            
            # 归一化（余弦相似度）
            faiss.normalize_L2(embeddings)
            
            # 构建FAISS索引（Flat索引，精确搜索）
            index = faiss.IndexFlatIP(self.dimension)  # 内积 = 余弦相似度（已归一化）
            index.add(embeddings)
            
            self.indices[cat_name] = index
            self.entries[cat_name] = entries
    
    def check(self, text: str, top_k: int = 3) -> Tuple[np.ndarray, List[Dict]]:
        """
        语义检测：返回四维分数 + 匹配详情
        分数：[品牌调性, 合规安全, 规范度, 综合语义]
        """
        model = _load_model()
        
        # 分句处理（简单按标点分割）
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not sentences:
            return np.array([0.0, 0.0, 1.0, 1.0]), []  # 无有效句子，默认规范
        
        # 编码所有句子
        sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
        sentence_embeddings = sentence_embeddings.astype('float32')
        faiss = _load_faiss()
        faiss.normalize_L2(sentence_embeddings)
        
        # 各类别检测
        results = {
            '口语化表达': {'max_sim': 0.0, 'matches': []},
            '敏感词风险': {'max_sim': 0.0, 'matches': []},
            '规范表达参考': {'max_sim': 0.0, 'matches': []}
        }
        
        all_matches = []
        
        for cat_name in ['口语化表达', '敏感词风险', '规范表达参考']:
            if cat_name not in self.indices:
                continue
            
            index = self.indices[cat_name]
            threshold = self.categories[cat_name]['threshold']
            entries = self.entries[cat_name]
            
            # FAISS检索：每个句子找最相似的top_k
            similarities, indices = index.search(sentence_embeddings, top_k)
            
            for sent_idx, (sims, idxs) in enumerate(zip(similarities, indices)):
                for sim, idx in zip(sims, idxs):
                    if sim > threshold and sim > results[cat_name]['max_sim']:
                        results[cat_name]['max_sim'] = float(sim)
                        
                        match_info = {
                            'category': cat_name,
                            'matched_sentence': sentences[sent_idx],
                            'matched_entry': entries[idx]['text'],
                            'similarity': float(sim),
                            'type': entries[idx]['type'],
                            'weight': entries[idx]['weight'],
                            'severity': self.categories[cat_name]['severity'],
                            'suggestion': self.categories[cat_name]['suggestion']
                        }
                        results[cat_name]['matches'].append(match_info)
                        all_matches.append(match_info)
        
        # 计算分数（归一化到0-1）
        # 口语化：越高越差
        口语化分数 = min(results['口语化表达']['max_sim'] * 1.2, 1.0)
        
        # 敏感风险：越高越差
        敏感分数 = min(results['敏感词风险']['max_sim'] * 1.1, 1.0)
        
        # 规范度：越高越好（与规范表达的相似度）
        规范分数 = results['规范表达参考']['max_sim']
        
        # 综合语义分：规范度 - 口语化 - 敏感风险（裁剪到0-1）
        综合分数 = max(0, 规范分数 - 口语化分数 * 0.5 - 敏感分数 * 0.5)
        
        scores = np.array([
            1.0 - 口语化分数,      # 品牌调性（1=好，0=差）
            1.0 - 敏感分数,        # 合规安全
            规范分数,              # 表达规范（参考）
            综合分数               # 综合语义
        ])
        
        return scores, all_matches
    
    def get_similar_words(self, query: str, category: str = None, top_k: int = 5) -> List[Dict]:
        """调试工具：查询与某个词最相似的库中条目"""
        model = _load_model()
        faiss = _load_faiss()
        
        query_vec = model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_vec)
        
        results = []
        cats = [category] if category else list(self.indices.keys())
        
        for cat in cats:
            if cat not in self.indices:
                continue
            sims, idxs = self.indices[cat].search(query_vec, top_k)
            for sim, idx in zip(sims[0], idxs[0]):
                results.append({
                    'category': cat,
                    'text': self.entries[cat][idx]['text'],
                    'similarity': float(sim),
                    'type': self.entries[cat][idx]['type']
                })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]


# 测试代码
if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("初始化语义检测器...")
    checker = SemanticChecker("config/synonyms_db.json")
    
    test_texts = [
        "亲，超赞的新书到了哦，赶紧来借吧！",  # 高口语化
        "免费下载知网论文，限时福利不要错过",  # 高敏感风险
        "图书馆提供学术资源服务，欢迎广大师生使用",  # 规范
        "这本书很棒，推荐大家阅读"  # 中等口语化
    ]
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"文本：{text}")
        scores, matches = checker.check(text)
        print(f"分数：[品牌调性={scores[0]:.3f}, 合规安全={scores[1]:.3f}, "
              f"规范度={scores[2]:.3f}, 综合语义={scores[3]:.3f}]")
        
        if matches:
            print(f"检测到 {len(matches)} 处语义匹配：")
            for m in matches[:3]:  # 只显示前3个
                print(f"  - [{m['category']}] '{m['matched_entry']}' "
                      f"(相似度{m['similarity']:.3f}, {m['type']})")