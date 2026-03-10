"""
text_utils.py
文本预处理工具函数
供 semantic_checker.py / fact_checker.py 共同调用
"""

import re
from typing import List, Dict

def split_sentences(text: str) -> List[str]:
    """
    分句处理：按中英文标点 + 逗号 + 换行符切割
    修复：原版不含逗号，长句无法切分导致向量质量下降
    """
    text = text.strip()
    # 中文句末标点 + 逗号（中英文）+ 换行
    sentences = re.split(r'[。！？；\n，,]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    return sentences

def clean_text(text: str) -> str:
    """
    文本清洗：合并空白符，过滤噪声字符
    修复：原版保留括号等噪声，影响向量编码质量
    """
    # 合并连续空白
    text = re.sub(r'\s+', ' ', text)
    # 保留中英文字符、数字、常用中文标点
    text = re.sub(r'[^\w\u4e00-\u9fff，。！？；：、]', ' ', text)
    return text.strip()

def extract_numbers(text: str) -> List[Dict]:
    """
    提取文本中所有数字表达（阿拉伯数字 + 中文数字），供 fact_checker 使用

    返回列表，每项格式：
    {
        'raw'  : 原始字符串（如 "二十" / "20"）,
        'value': 转换后的整数（如 20）,
        'start': 在原文中的起始位置,
        'end'  : 在原文中的结束位置
    }
    """
    results: List[Dict] = []

    # ── 阿拉伯数字（含小数）────────────────────────────────────────────
    for m in re.finditer(r'\d+(?:\.\d+)?', text):
        results.append({
            'raw':   m.group(),
            'value': float(m.group()),
            'start': m.start(),
            'end':   m.end(),
        })

    # ── 中文数字 → 阿拉伯数字映射 ──────────────────────────────────────
    CN_NUM = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '百': 100, '千': 1000, '万': 10000,
        '两': 2,
    }
    CN_UNIT = {'十', '百', '千', '万'}

    # 匹配纯中文数字串（如"二十""三百五十""两万"）
    cn_pattern = re.compile(r'[零一二三四五六七八九十百千万两]+')
    for m in cn_pattern.finditer(text):
        raw = m.group()
        # 跳过单个非数量词汉字误匹配（如"一般"中的"一"需结合上下文）
        # 简单策略：长度 >= 2 或 本身是量词单位才转换
        value = _cn_to_int(raw, CN_NUM, CN_UNIT)
        if value is not None:
            results.append({
                'raw':   raw,
                'value': float(value),
                'start': m.start(),
                'end':   m.end(),
            })

    # 按出现位置排序
    results.sort(key=lambda x: x['start'])
    return results

def _cn_to_int(cn_str: str, cn_map: Dict, unit_set: set):
    """
    将中文数字字符串转换为整数
    支持：二十、三百五十六、两万、十八 等常见表达
    不支持：亿级以上（图书馆场景无需）
    """
    if not cn_str:
        return None

    # 全为单位词（如"万"单独出现）→ 跳过
    if all(c in unit_set for c in cn_str):
        return None

    result  = 0
    current = 0   # 当前段累计值

    # 处理"十"开头的简写（如"十八" = 18）
    if cn_str[0] == '十':
        cn_str = '一' + cn_str

    for char in cn_str:
        if char not in cn_map:
            return None   # 含非数字字符，放弃转换
        val = cn_map[char]
        if val >= 10:
            # 遇到单位：将 current 乘以单位后累加
            current = current if current != 0 else 1
            result  += current * val
            current  = 0
        else:
            current = current * 10 + val

    result += current
    return result if result > 0 else None