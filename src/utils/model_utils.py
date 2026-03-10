"""
utils/model_utils.py
"""

import os

# ── 必须在所有 HF 相关 import 之前设置，否则不生效 ────────────────────────────
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HUGGINGFACE_HUB_VERBOSITY', 'warning')

import logging
import numpy as np
import faiss
import configparser

from pathlib import Path
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── 全局模型单例 ──────────────────────────────────────────────────────────────
_model:      SentenceTransformer | None = None
_model_name: str                        = ''
_embed_dim:  int                        = 512

# bge 系列 query 端前缀（passage 端不加）
_BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

def _is_bge(model_name: str) -> bool:
    """判断是否为 bge 系列模型"""
    return 'bge' in model_name.lower()

def _load_config() -> tuple[str, int]:
    """从 config.ini 读取模型名称和向量维度"""
    cfg      = configparser.ConfigParser()
    cfg_path = Path(__file__).resolve().parent.parent / 'config' / 'config.ini'

    if cfg_path.exists():
        cfg.read(cfg_path, encoding='utf-8')

    model_name = cfg.get('model', 'model_name', fallback='BAAI/bge-small-zh-v1.5')
    embed_dim  = cfg.getint('model', 'embedding_dim', fallback=512)
    return model_name, embed_dim

def get_model() -> SentenceTransformer:
    """
    全局模型单例，首次调用时从镜像源加载
    HF_ENDPOINT 已在模块顶部设为 hf-mirror.com
    """
    global _model, _model_name, _embed_dim

    if _model is not None:
        return _model

    _model_name, _embed_dim = _load_config()

    logger.info(
        f"[ModelUtils] 加载模型: {_model_name}  "
        f"镜像源: {os.environ.get('HF_ENDPOINT')}  "
        f"dim={_embed_dim}"
    )

    _model = SentenceTransformer(_model_name)
    logger.info(f"[ModelUtils] 模型加载完成: {_model_name}")
    return _model

def get_embedding_dim() -> int:
    """返回当前模型的向量维度"""
    global _embed_dim
    if not _embed_dim:
        _, _embed_dim = _load_config()
    return _embed_dim

def encode_queries(texts: list[str]) -> np.ndarray:
    """
    编码 query 端（待审核句子）
    bge 系列加 prefix，m3e 及其他直接 encode
    返回 L2 归一化 float32，shape=(n, dim)
    """
    model = get_model()

    if _is_bge(_model_name):
        texts = [_BGE_QUERY_PREFIX + t for t in texts]
        embs  = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
    else:
        embs = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        faiss.normalize_L2(embs)

    return embs

def encode_passages(texts: list[str]) -> np.ndarray:
    """
    编码 passage 端（词库词条）
    bge/m3e passage 端均不加前缀
    返回 L2 归一化 float32，shape=(n, dim)
    """
    model = get_model()

    embs = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=(_is_bge(_model_name)),
    ).astype(np.float32)

    if not _is_bge(_model_name):
        faiss.normalize_L2(embs)

    return embs

def build_faiss_index(texts: list[str], dimension: int | None = None) -> faiss.Index:
    """
    为词条列表构建 FAISS 内积索引
    内部使用 encode_passages 编码（passage 端不加 bge prefix）
    """
    dim  = dimension or get_embedding_dim()
    embs = encode_passages(texts)

    # 模型实际维度与配置不一致时，以模型为准并告警
    if embs.shape[1] != dim:
        logger.warning(
            f"[ModelUtils] 维度不匹配: 模型={embs.shape[1]}  config={dim}  "
            f"自动使用模型实际维度"
        )
        dim = embs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    logger.debug(f"[ModelUtils] FAISS 索引构建完成: {index.ntotal} 条  dim={dim}")
    return index