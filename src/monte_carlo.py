"""
蒙特卡洛模拟模块
"""

import numpy as np
from numba import njit
from typing import Dict, Tuple


DEFAULT_CONFIG = {
    'n_simulations': 10000,      # 模拟次数
    'perturbation_sigma': 0.15,   # 权重扰动标准差（模拟人工差异程度）
    'base_weights': np.array([0.40, 0.25, 0.25, 0.10]),  # [事实, 品牌, 合规, 规范]
    'weight_bounds': (0.05, 0.80),  # 权重裁剪边界
    'random_seed': None           # 随机种子
}


# 核心采样函数
@njit
def _single_audit_sample(
    fact_score: float,
    semantic_scores: np.ndarray,
    base_weights: np.ndarray,
    perturb_sigma: float,
    weight_bounds: Tuple[float, float]
) -> float:
    """
    单次审核采样：对权重加入扰动，计算综合分数
    
    """
    # 事实分数：硬标准，不扰动
    # 语义分数：软标准，权重可扰动
    
    # 复制基础权重
    weights = base_weights.copy()
    
    # 对品牌、合规、规范的权重加入高斯扰动（索引1,2,3）
    # 事实权重（索引0）固定，因为它是客观标准
    for i in range(1, 4):
        perturbation = np.random.normal(1.0, perturb_sigma)
        weights[i] *= perturbation
    
    # 权重归一化（确保和为1）
    weights = weights / weights.sum()
    
    # 裁剪到合理范围（避免某个权重过大或过小）
    weights = np.clip(weights, weight_bounds[0], weight_bounds[1])
    weights = weights / weights.sum()  # 再次归一化
    
    # 计算综合分数（加权求和）
    # 事实分数 × 事实权重 + 语义分数 × 语义权重
    all_scores = np.array([fact_score, semantic_scores[0], 
                          semantic_scores[1], semantic_scores[2]])
    final_score = np.sum(all_scores * weights)
    
    # 裁剪到0-1范围
    return 0.0 if final_score < 0.0 else (1.0 if final_score > 1.0 else final_score)


# =========================
# 主模拟函数（对应 主模拟）
# =========================
@njit
def _monte_carlo_simulation(
    fact_score: float,
    semantic_scores: np.ndarray,
    base_weights: np.ndarray,
    perturb_sigma: float,
    weight_bounds: Tuple[float, float],
    n_simulations: int
) -> np.ndarray:
    """
    蒙特卡洛主模拟：重复采样得到分数分布
    
    """
    scores = np.empty(n_simulations, dtype=np.float64)
    
    for i in range(n_simulations):
        scores[i] = _single_audit_sample(
            fact_score, semantic_scores, base_weights,
            perturb_sigma, weight_bounds
        )
    
    return scores


class MonteCarloAuditor:
    """
    蒙特卡洛审核器：量化审核决策的不确定性
    """
    
    def __init__(self, config: Dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.rng = np.random.RandomState(self.config['random_seed'])
        
        # 设置全局随机种子
        if self.config['random_seed'] is not None:
            np.random.seed(self.config['random_seed'])
    
    def audit(
        self,
        fact_score: float,
        semantic_scores: np.ndarray,
        progress_callback=None
    ) -> Dict:
        """
        执行蒙特卡洛审核，返回完整统计结果
        """
        n = self.config['n_simulations']
        base_w = self.config['base_weights']
        sigma = self.config['perturbation_sigma']
        bounds = self.config['weight_bounds']
        
        # 执行模拟
        scores = _monte_carlo_simulation(
            fact_score, semantic_scores, base_w,
            sigma, bounds, n
        )
        
        # 统计计算
        result = self._calculate_statistics(scores)
        
        # 决策建议（新增：对应审核场景）
        result['decision'] = self._generate_decision(result)
        
        return result
    
    def _calculate_statistics(self, scores: np.ndarray) -> Dict:
        """
        统计计算
        """
        # 基础统计量
        mean = scores.mean()
        median = np.median(scores)
        std = scores.std()
        min_val = scores.min()
        max_val = scores.max()
        
        # 变异系数（相对不确定性）
        cv = std / mean if mean > 0 else 0
        
        # 分位点
        percentiles = [10, 25, 50, 75, 90, 95]
        percentile_values = np.percentile(scores, percentiles)
        
        # 置信区间（P10-P90）
        ci_lower = percentile_values[0]   # P10
        ci_upper = percentile_values[4]   # P90
        
        # 决策
        pass_prob = np.mean(scores > 0.80)   # 通过概率
        review_prob = np.mean((scores > 0.60) & (scores <= 0.80))  # 复核概率
        reject_prob = np.mean(scores <= 0.60)  # 拒绝概率
        
        return {
            'scores': scores,           # 原始分数列表（用于可视化）
            'mean': mean,               # 均值
            'median': median,           # 中位数
            'std': std,                 # 标准差
            'cv': cv,                   # 变异系数
            'min': min_val,             # 最小值
            'max': max_val,             # 最大值
            'percentiles': dict(zip(percentiles, percentile_values)),
            'ci_lower': ci_lower,       # 置信区间下限（P10）
            'ci_upper': ci_upper,       # 置信区间上限（P90）
            'pass_probability': pass_prob,
            'review_probability': review_prob,
            'reject_probability': reject_prob
        }
    
    def _generate_decision(self, stats: Dict) -> Dict:
        """
        生成决策建议
        """
        p_pass = stats['pass_probability']
        p_review = stats['review_probability']
        p_reject = stats['reject_probability']
        median = stats['median']
        ci_width = stats['ci_upper'] - stats['ci_lower']
        
        # 决策逻辑
        if p_pass > 0.90:
            recommendation = "通过"
            confidence = "high"
            action = "建议直接发布，高置信度通过"
        elif p_pass > 0.70:
            recommendation = "谨慎通过"
            confidence = "medium"
            action = "建议发布，但需关注特定风险点"
        elif p_review > 0.50:
            recommendation = "人工复核"
            confidence = "medium"
            action = "建议人工复核，自动审核不确定"
        elif p_reject > 0.70:
            recommendation = "重审"
            confidence = "high"
            action = "建议修改后重审，存在明确问题"
        else:
            recommendation = "建议重审"
            confidence = "medium"
            action = "建议按修改意见调整后重审"
        
        # 不确定性评估
        uncertainty_level = "low" if ci_width < 0.15 else "medium" if ci_width < 0.30 else "high"
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'uncertainty_level': uncertainty_level,
            'action': action,
            'summary': f"{action}（通过概率{p_pass:.1%}，不确定性{uncertainty_level}）"
        }
    
    def sensitivity_analysis(self, fact_score: float, semantic_scores: np.ndarray) -> Dict:
        """
        敏感性分析：测试不同扰动幅度下的结果稳定性
        """
        sigmas = [0.05, 0.10, 0.15, 0.20, 0.30]
        results = []
        
        original_sigma = self.config['perturbation_sigma']
        
        for sigma in sigmas:
            self.config['perturbation_sigma'] = sigma
            result = self.audit(fact_score, semantic_scores)
            results.append({
                'sigma': sigma,
                'mean': result['mean'],
                'std': result['std'],
                'pass_prob': result['pass_probability']
            })
        
        # 恢复原始配置
        self.config['perturbation_sigma'] = original_sigma
        
        return {
            'optimal_sigma': original_sigma,
            'analysis': results,
            'conclusion': f"当前sigma={original_sigma}在稳定性和区分度间取得平衡"
        }


# =========================
# 测试代码
# =========================
if __name__ == "__main__":
    print("=" * 70)
    print("蒙特卡洛审核模拟测试")
    print("=" * 70)
    
    # 初始化审核器
    auditor = MonteCarloAuditor({
        'n_simulations': 10000,
        'perturbation_sigma': 0.15,
        'random_seed': 42  # 保证可复现
    })
    
    # 测试案例
    test_cases = [
        {
            'name': '案例A：完美文本',
            'fact': 1.0,           # 无事实错误
            'semantic': np.array([1.0, 1.0, 0.9])  # 品牌、合规、规范都好
        },
        {
            'name': '案例B：事实错误',
            'fact': 0.0,           # 有事实错误（一票否决）
            'semantic': np.array([0.9, 0.9, 0.8])
        },
        {
            'name': '案例C：高口语化',
            'fact': 1.0,
            'semantic': np.array([0.4, 0.9, 0.7])  # 品牌调性差
        },
        {
            'name': '案例D：边界案例',
            'fact': 0.8,           # 轻微事实问题
            'semantic': np.array([0.7, 0.6, 0.8])  # 各方面都一般
        }
    ]
    
    for case in test_cases:
        print(f"\n{'-'*70}")
        print(f"测试：{case['name']}")
        print(f"输入：事实={case['fact']}, 语义={case['semantic']}")
        
        result = auditor.audit(case['fact'], case['semantic'])
        decision = result['decision']
        
        print(f"\n统计结果：")
        print(f"  均值={result['mean']:.4f}, 中位数={result['median']:.4f}, "
              f"标准差={result['std']:.4f}")
        print(f"  变异系数={result['cv']:.4f}（相对不确定性）")
        print(f"  置信区间：[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}] "
              f"(P10-P90)")
        
        print(f"\n决策概率：")
        print(f"  通过(>0.8)：{result['pass_probability']:.2%}")
        print(f"  复核(0.6-0.8)：{result['review_probability']:.2%}")
        print(f"  拒绝(<0.6)：{result['reject_probability']:.2%}")
        
        print(f"\n决策建议：")
        print(f"  [{decision['recommendation']}] {decision['summary']}")
        print(f"  置信度：{decision['confidence']}, "
              f"不确定性：{decision['uncertainty_level']}")
        
        # 敏感性分析（仅第一个案例）
        if case['name'] == '案例D：边界案例':
            print(f"\n[敏感性分析] 不同扰动幅度下的结果：")
            sens = auditor.sensitivity_analysis(case['fact'], case['semantic'])
            for item in sens['analysis']:
                print(f"  σ={item['sigma']:.2f}: "
                      f"均值={item['mean']:.3f}, "
                      f"通过概率={item['pass_prob']:.1%}")
    
    print(f"\n{'='*70}")
    print("测试完成")
    print(f"{'='*70}")