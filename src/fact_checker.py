"""
事实核查模块 - 基于正则规则库
"""

import json
import re
from typing import List, Dict, Tuple
import numpy as np


class FactChecker:
    def __init__(self, db_path: str = "config/facts_db.json"):
        with open(db_path, 'r', encoding='utf-8') as f:
            self.db = json.load(f)
        self.rules = self.db['rules']
    
    def check(self, text: str) -> Tuple[float, List[Dict]]:
        """
        事实核查：返回分数(0或1) + 错误详情列表
        分数：1.0 = 无事实错误，0.0 = 有事实错误（一票否决）
        """
        errors = []
        
        for rule in self.rules:
            result = self._apply_rule(text, rule)
            if result['matched'] and result['is_error']:
                errors.append({
                    'rule_id': rule['id'],
                    'rule_name': rule['name'],
                    'severity': rule['severity'],
                    'message': result['message'],
                    'suggestion': rule.get('suggestion', ''),
                    'matched_text': result['matched_text']
                })
        
        # 事实核查是硬标准：有error级错误则0分，只有warning则0.5分，无错误1分
        if any(e['severity'] == 'error' for e in errors):
            score = 0.0
        elif errors:  # 只有warning
            score = 0.5
        else:
            score = 1.0
            
        return score, errors
    
    def _apply_rule(self, text: str, rule: Dict) -> Dict:
        """应用单条规则"""
        result = {
            'matched': False,
            'is_error': False,
            'message': '',
            'matched_text': ''
        }
        
        pattern_type = rule['pattern_type']
        pattern = rule['pattern']
        
        # 查找匹配
        matches = list(re.finditer(pattern, text))
        if not matches:
            return result
        
        result['matched'] = True
        result['matched_text'] = matches[0].group(0)
        
        # 根据规则类型判断是否为错误
        if pattern_type == 'regex':
            result['is_error'] = self._check_regex_rule(text, rule, matches)
        elif pattern_type == 'regex_number_range':
            result['is_error'] = self._check_number_range(text, rule, matches)
        
        if result['is_error']:
            result['message'] = self._format_error_message(rule, result['matched_text'])
        
        return result
    
    def _check_regex_rule(self, text: str, rule: Dict, matches: List) -> bool:
        """正则规则：检查是否在invalid_values中，或不在valid_values中"""
        matched_text = matches[0].group(0)
        
        # 优先检查invalid_values（明确错误）
        if 'invalid_values' in rule:
            for invalid in rule['invalid_values']:
                if invalid in matched_text or invalid in text:
                    return True
        
        # 再检查valid_values（不在其中则警告）
        if 'valid_values' in rule:
            for valid in rule['valid_values']:
                if valid in matched_text or valid in text:
                    return False  # 找到有效值，不是错误
            # 没找到有效值，但匹配了模式，可能是错误
            return rule.get('severity') == 'error'
        
        return False
    
    def _check_number_range(self, text: str, rule: Dict, matches: List) -> bool:
        """数字范围规则：提取数字，检查是否超限"""
        max_value = rule['max_value']
        
        for match in matches:
            try:
                number = int(match.group(1))
                if number > max_value:
                    return True
            except (ValueError, IndexError):
                continue
        
        return False
    
    def _format_error_message(self, rule: Dict, matched_text: str) -> str:
        """格式化错误信息"""
        message = rule['error_message']
        if '{value}' in message:
            # 尝试提取数字
            numbers = re.findall(r'\d+', matched_text)
            if numbers:
                message = message.replace('{value}', numbers[0])
        return message


# 测试代码
if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    checker = FactChecker("config/facts_db.json")
    
    test_texts = [
        "图书馆明天开放到23点，欢迎大家来借50本书！",  # 2个错误
        "总馆开放时间为8:00-22:00，本科生可借20册",  # 正确
        "联系电话：0431-85161234"  # warning
    ]
    
    for text in test_texts:
        score, errors = checker.check(text)
        print(f"\n文本：{text[:30]}...")
        print(f"分数：{score}")
        if errors:
            for e in errors:
                print(f"  - [{e['severity']}] {e['rule_name']}: {e['message']}")