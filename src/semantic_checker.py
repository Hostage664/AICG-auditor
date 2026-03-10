"""
semantic_checker.py
"""

import json
import logging
import numpy as np
import faiss as _faiss

from pathlib import Path
from typing  import Dict, List, Tuple, Optional

# 修改：导入 encode_queries，不再直接调用 get_model().encode()
from utils.model_utils import encode_queries, build_faiss_index
from utils.text_utils  import split_sentences, clean_text

logger = logging.getLogger(__name__)

_BLACKLIST_SIM_FLOOR = 0.60   # bge/m3e 中文语义清晰，阈值可恢复正常
_WHITELIST_SIM_FLOOR = 0.45

class _DBIndex:
    """单份 DB 的 FAISS 索引封装"""

    def __init__(self, db_path: str, db_role: str):
        self.db_role = db_role
        self.db_path = str(Path(db_path).resolve())

        with open(self.db_path, encoding='utf-8') as f:
            raw = json.load(f)

        self.dimension: int = raw.get('metadata', {}).get('dimension', 512)
        self._categories: Dict[str, List[Dict]] = {}

        if 'categories' in raw:
            for name, body in raw['categories'].items():
                entries = body.get('entries', [])
                if entries:
                    self._categories[name] = entries
        elif 'entries' in raw:
            self._categories['default'] = raw['entries']
        else:
            raise ValueError(f"[_DBIndex:{db_role}] {self.db_path} 格式不合法")

        self._indices: Dict[str, object] = {}
        for cat_name, entries in self._categories.items():
            texts = [e['text'] for e in entries]
            logger.info(f"[_DBIndex:{db_role}] [{cat_name}] 构建索引: {len(texts)} 条")
            # passage 端由 build_faiss_index 内部使用 encode_passages 编码
            self._indices[cat_name] = build_faiss_index(texts, self.dimension)

    def search(
        self,
        sent_embs: np.ndarray,   # 已由 encode_queries 处理，shape=(n_sents, dim)
        sentences: List[str],
        top_k:     int   = 3,
        sim_floor: float = 0.0,
    ) -> List[Dict]:
        results: List[Dict] = []

        for cat_name, index in self._indices.items():
            entries    = self._categories[cat_name]
            k          = min(top_k, index.ntotal)
            sims, idxs = index.search(sent_embs, k)

            for sent_i, (sim_row, idx_row) in enumerate(zip(sims, idxs)):
                for sim_val, entry_idx in zip(sim_row, idx_row):
                    if entry_idx < 0:
                        continue
                    sim_f = float(sim_val)
                    if sim_f < sim_floor:
                        continue

                    entry = entries[entry_idx]
                    w     = float(entry.get('weight', 1.0))
                    results.append({
                        'category':   cat_name,
                        'sentence':   sentences[sent_i],
                        'matched':    entry['text'],
                        'similarity': sim_f,
                        'weight':     w,
                        'w_sim':      sim_f * w,
                        'db_role':    self.db_role,
                    })

        return results

class SemanticChecker:
    """
    双库语义检测器
    """

    _cache: Dict[str, _DBIndex] = {}

    def __init__(
        self,
        blacklist_db_path:        str,
        whitelist_db_path:        str,
        blacklist_category_roles: Optional[Dict[str, str]] = None,
    ):
        self.blacklist_path = str(Path(blacklist_db_path).resolve())
        self.whitelist_path = str(Path(whitelist_db_path).resolve())
        self._cat_roles     = blacklist_category_roles or {}

        if self.blacklist_path not in SemanticChecker._cache:
            SemanticChecker._cache[self.blacklist_path] = _DBIndex(
                self.blacklist_path, 'blacklist'
            )
        if self.whitelist_path not in SemanticChecker._cache:
            SemanticChecker._cache[self.whitelist_path] = _DBIndex(
                self.whitelist_path, 'whitelist'
            )

        self._bl: _DBIndex = SemanticChecker._cache[self.blacklist_path]
        self._wl: _DBIndex = SemanticChecker._cache[self.whitelist_path]

    def check(
        self,
        text:  str,
        top_k: int = 3,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        双库检测
        scores[0] brand_score      = 1 - mean(口语化惩罚池)
        scores[1] compliance_score = 1 - mean(敏感词惩罚池)
        scores[2] norm_score       = mean(白名单奖励池)
        """
        if not text or not text.strip():
            logger.warning("[SemanticChecker] 空文本，返回保守默认分")
            return np.array([1.0, 1.0, 0.0], dtype=np.float64), []

        sentences = split_sentences(clean_text(text)) or [text.strip()]

        sent_embs = encode_queries(sentences)

        bl_matches = self._bl.search(
            sent_embs, sentences,
            top_k=top_k, sim_floor=_BLACKLIST_SIM_FLOOR,
        )
        wl_matches = self._wl.search(
            sent_embs, sentences,
            top_k=top_k, sim_floor=_WHITELIST_SIM_FLOOR,
        )

        brand_pen:      List[float] = []
        compliance_pen: List[float] = []

        for m in bl_matches:
            role = self._cat_roles.get(m['category'], 'brand')
            (compliance_pen if role == 'compliance' else brand_pen).append(m['w_sim'])

        norm_pool = [m['w_sim'] for m in wl_matches]

        brand_score      = float(np.clip(1.0 - _safe_mean(brand_pen),      0.0, 1.0))
        compliance_score = float(np.clip(1.0 - _safe_mean(compliance_pen), 0.0, 1.0))
        norm_score       = float(np.clip(_safe_mean(norm_pool),            0.0, 1.0))

        scores = np.array(
            [brand_score, compliance_score, norm_score],
            dtype=np.float64,
        )

        logger.info(
            f"[SemanticChecker] "
            f"bl_hits={len(bl_matches)}  wl_hits={len(wl_matches)}\n"
            f"  brand={brand_score:.3f}  "
            f"compliance={compliance_score:.3f}  "
            f"norm={norm_score:.3f}"
        )

        return scores, bl_matches + wl_matches

def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0