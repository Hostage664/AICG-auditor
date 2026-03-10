"""
fact_checker.py
事实核查模块
规则库字段说明：
  pattern_type（推荐）或 type（兼容旧版）：regex / regex_number_range / phone_format
"""

import json
import re
import logging
from pathlib import Path
from typing  import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 规则类型 → 必填字段
_REQUIRED_FIELDS: Dict[str, List[str]] = {
    'regex':              ['id', 'name', 'pattern_type', 'pattern', 'severity', 'error_message'],
    'regex_number_range': ['id', 'name', 'pattern_type', 'pattern', 'severity', 'error_message'],
    'phone_format':       ['id', 'name', 'pattern_type',             'severity', 'error_message'],
}

_VALID_TYPES = set(_REQUIRED_FIELDS.keys())

# 电话号码正则
_PHONE_LOOSE  = re.compile(r'\d{3,4}[-\s]?\d{7,8}|\d{11}')
_PHONE_STRICT = re.compile(r'^(0\d{2,3}-\d{7,8}|1[3-9]\d{9})$')

class FactChecker:
    """
    事实核查器：加载 facts_db.json → 预编译正则 → 结构化返回审核结果
    支持三种规则类型：regex / regex_number_range / phone_format
    字段兼容：'pattern_type' 优先，退而检查旧版 'type' 字段
    """

    def __init__(self, db_path: str = 'config/facts_db.json'):
        path = Path(db_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"[FactChecker] 规则库不存在: {path}")

        with open(str(path), encoding='utf-8') as f:
            raw = json.load(f)

        if not isinstance(raw.get('rules'), list):
            raise ValueError("[FactChecker] facts_db.json 缺少有效的 'rules' 列表")

        self.rules: List[Dict] = []
        self._compiled: Dict[str, re.Pattern] = {}
        skipped = 0

        for i, rule in enumerate(raw['rules']):
            # ── 兼容 'type' 和 'pattern_type' 两种字段名 ──────────────────
            p_type = (
                rule.get('pattern_type') or   # 优先使用 pattern_type
                rule.get('type') or            # 兼容旧版 type 字段
                ''
            ).strip()

            if not p_type:
                logger.warning(
                    f"[FactChecker] 规则[{i}] '{rule.get('name', '未命名')}' "
                    f"缺少 'pattern_type' 字段 → 跳过\n"
                    f"  合法值: {_VALID_TYPES}\n"
                    f"  请在 facts_db.json 第 {i+1} 条规则中添加 \"pattern_type\" 字段"
                )
                skipped += 1
                continue

            if p_type not in _VALID_TYPES:
                logger.warning(
                    f"[FactChecker] 规则[{i}] '{rule.get('name', '未命名')}' "
                    f"pattern_type='{p_type}' 不在合法值集合 {_VALID_TYPES} 中 → 跳过"
                )
                skipped += 1
                continue

            # 统一写回 pattern_type，消除后续对字段名的歧义
            rule['pattern_type'] = p_type

            # 预编译正则
            if p_type in ('regex', 'regex_number_range'):
                pattern = rule.get('pattern', '')
                if not pattern:
                    logger.warning(
                        f"[FactChecker] 规则[{i}] '{rule.get('name')}' "
                        f"缺少 'pattern' 字段 → 跳过"
                    )
                    skipped += 1
                    continue
                try:
                    self._compiled[rule['id']] = re.compile(pattern)
                except re.error as e:
                    logger.warning(
                        f"[FactChecker] 规则[{i}] '{rule.get('name')}' "
                        f"正则编译失败: {e} → 跳过"
                    )
                    skipped += 1
                    continue

            self.rules.append(rule)

        logger.info(
            f"[FactChecker] 加载 {len(self.rules)} 条规则"
            f"（跳过: {skipped}） ← {path}"
        )

    # ── 公开接口 ──────────────────────────────────────────────────────────────
    def check(self, text: str) -> Tuple[float, List[Dict]]:
        """
        执行所有规则检查
        返回 (fact_score, issues)
          fact_score : error 级问题存在 → 0.0，否则 max(0, 1.0 - issues*0.1)
          issues     : 命中的规则问题列表
        """
        issues: List[Dict] = []

        for rule in self.rules:
            try:
                issue = self._apply_rule(text, rule)
                if issue:
                    issues.append(issue)
            except Exception as e:
                logger.warning(
                    f"[FactChecker] 规则 '{rule.get('name', '?')}' 执行异常: {e}"
                )

        has_error  = any(i.get('severity') == 'error' for i in issues)
        fact_score = 0.0 if has_error else max(0.0, 1.0 - len(issues) * 0.1)

        return fact_score, issues

    # ── 规则分发 ──────────────────────────────────────────────────────────────
    def _apply_rule(self, text: str, rule: Dict) -> Optional[Dict]:
        p_type = rule['pattern_type']
        if p_type == 'regex':
            return self._check_regex(text, rule)
        if p_type == 'regex_number_range':
            return self._check_number_range(text, rule)
        if p_type == 'phone_format':
            return self._check_phone(text, rule)
        return None

    # ── regex ─────────────────────────────────────────────────────────────────
    def _check_regex(self, text: str, rule: Dict) -> Optional[Dict]:
        """命中正则 → 视为问题"""
        pat = self._compiled.get(rule['id'])
        if pat is None or not pat.search(text):
            return None
        return _make_issue(rule, f"命中规则 '{rule['name']}'")

    # ── regex_number_range ────────────────────────────────────────────────────
    def _check_number_range(self, text: str, rule: Dict) -> Optional[Dict]:
        """从文本提取数值，检查是否落在 [min_val, max_val] 范围内"""
        pat = self._compiled.get(rule['id'])
        if pat is None:
            return None

        min_val = _to_float(rule.get('min_val'))
        max_val = _to_float(rule.get('max_val'))
        if min_val is None and max_val is None:
            return None

        violations: List[str] = []
        for match in pat.finditer(text):
            # 优先取第一个捕获组，无捕获组则取整体
            raw = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
            num = _to_float(raw)
            if num is None:
                continue
            if min_val is not None and num < min_val:
                violations.append(f"{num} < 最小值 {min_val}")
            if max_val is not None and num > max_val:
                violations.append(f"{num} > 最大值 {max_val}")

        if not violations:
            return None

        return _make_issue(rule, '；'.join(violations))

    # ── phone_format ──────────────────────────────────────────────────────────
    def _check_phone(self, text: str, rule: Dict) -> Optional[Dict]:
        """检查文本中所有电话号码的格式是否规范"""
        phones = _PHONE_LOOSE.findall(text)
        bad    = [p for p in phones if not _PHONE_STRICT.match(p.replace(' ', ''))]

        if not bad:
            return None

        return _make_issue(rule, f"格式不规范的电话号码: {bad}")

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _make_issue(rule: Dict, detail: str) -> Dict:
    """构造统一的 issue 字典"""
    return {
        'rule_id':   rule.get('id', ''),
        'rule_name': rule.get('name', ''),
        'severity':  rule.get('severity', 'warning'),
        'message':   rule.get('error_message', detail),
        'detail':    detail,
    }

def _to_float(value) -> Optional[float]:
    """安全转换为 float，失败返回 None"""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None