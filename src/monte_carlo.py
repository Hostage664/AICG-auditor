"""
monte_carlo.py
蒙特卡洛综合评分审核模块
接口：MonteCarloAuditor(config: dict)
      config 必须包含 base_weights / perturbation_sigma / weight_bounds /
                       n_simulations / pass_threshold / review_threshold
      可选包含 random_seed / single_dim_reject / bc_avg_reject /
               fact_review_floor / pass_prob_min（硬性规则阈值）
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── 默认配置（仅作兜底，实际由 config.ini 通过 batch_audit 传入）────────────
_DEFAULT_CONFIG: Dict[str, Any] = {
    'base_weights':       [0.50, 0.18, 0.20, 0.12],
    'perturbation_sigma': 0.05,
    'weight_bounds':      (0.05, 0.75),
    'n_simulations':      10_000,
    'pass_threshold':     0.82,
    'review_threshold':   0.60,
    'random_seed':        42,
    # 硬性否决阈值
    'single_dim_reject':  0.45,
    'bc_avg_reject':      0.50,
    # 硬性复核阈值
    'fact_review_floor':  0.95,
    'pass_prob_min':      0.60,
}

# config 必须包含的字段
_REQUIRED_KEYS: List[str] = [
    'base_weights', 'perturbation_sigma', 'weight_bounds',
    'n_simulations', 'pass_threshold', 'review_threshold',
]

class MonteCarloAuditor:

    def __init__(self, config: dict = None):
        """
        初始化蒙特卡洛审核器
        :param config: 由 batch_audit._load_config() 传入的配置 dict
                       传 None 时使用内置默认值并打印警告
        """
        if config is None:
            logger.warning(
                "[MonteCarloAuditor] 未传入外部配置，使用内置默认值\n"
                "  请检查 batch_audit.py 是否正确读取并传入 config.ini"
            )
            cfg = dict(_DEFAULT_CONFIG)
        else:
            # 检查必要字段
            missing = [k for k in _REQUIRED_KEYS if k not in config]
            if missing:
                raise KeyError(
                    f"[MonteCarloAuditor] 传入 config 缺少必要字段: {missing}"
                )
            # 外部配置覆盖默认值（保留硬性规则字段的默认值）
            cfg = {**_DEFAULT_CONFIG, **config}

        # base_weights：统一转为 np.ndarray 并归一化
        # 修复核心：无论传入 list / np.ndarray / 其他可迭代，统一用 np.asarray
        #           不用 np.array(weights) ← 旧版写法，传入 dict 时直接崩溃
        cfg['base_weights'] = np.asarray(
            cfg['base_weights'], dtype=np.float64
        ).ravel()

        w_sum = cfg['base_weights'].sum()
        if w_sum <= 0:
            raise ValueError("[MonteCarloAuditor] base_weights 之和必须 > 0")
        cfg['base_weights'] = cfg['base_weights'] / w_sum

        cfg['weight_bounds'] = tuple(cfg['weight_bounds'])

        self.config = cfg
        self._rng   = np.random.default_rng(int(cfg['random_seed']))

        logger.info(
            f"[MonteCarloAuditor] 初始化完成\n"
            f"  weights        = {np.round(cfg['base_weights'], 3).tolist()}\n"
            f"  n_simulations  = {cfg['n_simulations']}\n"
            f"  pass_threshold = {cfg['pass_threshold']}  "
            f"review_threshold = {cfg['review_threshold']}\n"
            f"  hard_reject: single<{cfg['single_dim_reject']} "
            f"| bc_avg<{cfg['bc_avg_reject']}\n"
            f"  hard_review: fact<{cfg['fact_review_floor']} "
            f"| pass_prob<{cfg['pass_prob_min']}"
        )

    # ── 主入口 ────────────────────────────────────────────────────────────────
    def audit(
        self,
        fact_score:   float,
        sem_scores:   np.ndarray,  # shape=(3,): [brand, compliance, norm]
    ) -> Dict[str, Any]:
        """
        执行蒙特卡洛模拟并作出审核决策
        :param fact_score:  事实核查得分 [0,1]
        :param sem_scores:  语义检测三维得分 [brand, compliance, norm]
        :return: 包含 mean/median/std/ci_95/pass_probability/
                 review_probability/reject_probability/decision 的字典
        """
        sem_scores = np.asarray(sem_scores, dtype=np.float64).ravel()
        if sem_scores.shape[0] != 3:
            raise ValueError(
                f"[MonteCarloAuditor] sem_scores 期望 shape=(3,)，"
                f"收到 shape={sem_scores.shape}"
            )

        # 拼接四维得分向量 [fact, brand, compliance, norm]
        all_scores = np.concatenate([[float(fact_score)], sem_scores])

        # 执行模拟
        mc_stats = self._simulate(all_scores)

        # 决策
        decision = self._make_decision(
            mc_result  = mc_stats,
            fact_score = float(fact_score),
            sem_scores = sem_scores,
        )

        return {**mc_stats, 'decision': decision}

    # ── 蒙特卡洛模拟 ─────────────────────────────────────────────────────────
    def _simulate(self, scores: np.ndarray) -> Dict[str, float]:
        """对权重加高斯扰动，统计加权得分分布"""
        cfg  = self.config
        n    = int(cfg['n_simulations'])
        base = cfg['base_weights'].copy()         # shape=(4,)
        w_lo, w_hi = cfg['weight_bounds']
        sigma      = float(cfg['perturbation_sigma'])

        # 权重扰动并归一化
        noise       = self._rng.normal(0, sigma, size=(n, len(base)))
        w_perturbed = np.clip(base + noise, w_lo, w_hi)
        w_perturbed = w_perturbed / w_perturbed.sum(axis=1, keepdims=True)  # shape=(n,4)

        # 加权得分
        sim_scores = w_perturbed @ scores   # shape=(n,)

        pass_thr   = float(cfg['pass_threshold'])
        review_thr = float(cfg['review_threshold'])

        ci_low, ci_high = np.percentile(sim_scores, [2.5, 97.5])

        return {
            'mean':               float(np.mean(sim_scores)),
            'median':             float(np.median(sim_scores)),
            'std':                float(np.std(sim_scores)),
            'ci_95':              [float(ci_low), float(ci_high)],
            'pass_probability':   float(np.mean(sim_scores >= pass_thr)),
            'review_probability': float(np.mean(
                (sim_scores >= review_thr) & (sim_scores < pass_thr)
            )),
            'reject_probability': float(np.mean(sim_scores < review_thr)),
        }

    # ── 决策逻辑 ──────────────────────────────────────────────────────────────
    def _make_decision(
        self,
        mc_result:  Dict[str, float],
        fact_score: float,
        sem_scores: np.ndarray,   # [brand, compliance, norm]
    ) -> Dict[str, Any]:
        """
        决策优先级（从高到低）：
          ① 硬性否决 — brand/compliance 严重偏低 → REJECT
          ② 硬性复核 — fact_score 有问题 / pass_prob 不足 → REVIEW
          ③ 概率阈值 — 正常流程
        """
        cfg        = self.config
        brand      = float(sem_scores[0])
        compliance = float(sem_scores[1])
        mean       = mc_result['mean']
        pass_prob  = mc_result['pass_probability']

        # ── ① 硬性否决 ────────────────────────────────────────────────────────
        single_thr = float(cfg['single_dim_reject'])
        avg_thr    = float(cfg['bc_avg_reject'])

        if brand < single_thr:
            return _reject(
                f"品牌调性分 {brand:.3f} < 否决阈值 {single_thr}，"
                f"内容风格严重不符，建议重写",
                trigger='hard_reject',
            )
        if compliance < single_thr:
            return _reject(
                f"合规安全分 {compliance:.3f} < 否决阈值 {single_thr}，"
                f"存在敏感内容风险，建议重写",
                trigger='hard_reject',
            )
        if (brand + compliance) / 2 < avg_thr:
            return _reject(
                f"品牌+合规均值 {(brand+compliance)/2:.3f} < {avg_thr}，"
                f"整体内容质量不足，建议重写",
                trigger='hard_reject',
            )

        # ── ② 硬性复核 ────────────────────────────────────────────────────────
        fact_floor    = float(cfg['fact_review_floor'])
        pass_prob_min = float(cfg['pass_prob_min'])

        if fact_score < fact_floor:
            return _review(
                f"事实核查分 {fact_score:.3f} < {fact_floor}，"
                f"存在事实性问题，需人工核实",
                trigger='hard_review',
            )
        if pass_prob < pass_prob_min:
            return _review(
                f"通过概率 {pass_prob:.2%} < {pass_prob_min:.0%}，"
                f"置信度不足，需人工确认",
                trigger='hard_review',
            )

        # ── ③ 概率阈值判断 ────────────────────────────────────────────────────
        pass_thr   = float(cfg['pass_threshold'])
        review_thr = float(cfg['review_threshold'])

        if mean >= pass_thr:
            return _approve(
                f"均值 {mean:.3f} ≥ 通过阈值 {pass_thr}",
                trigger='probability',
            )
        if mean >= review_thr:
            return _review(
                f"均值 {mean:.3f} 处于复核区间 [{review_thr}, {pass_thr})，"
                f"建议人工确认",
                trigger='probability',
            )
        return _reject(
            f"均值 {mean:.3f} < 复核阈值 {review_thr}，内容质量不足",
            trigger='probability',
        )

# ── 决策构造辅助函数 ──────────────────────────────────────────────────────────

def _approve(reason: str, trigger: str) -> Dict[str, Any]:
    return {
        'recommendation': '建议通过',
        'action':         'APPROVE',
        'confidence':     'HIGH',
        'uncertainty_level': 'LOW',
        'reason':         reason,
        'trigger':        trigger,
    }

def _review(reason: str, trigger: str) -> Dict[str, Any]:
    return {
        'recommendation': '建议人工复核',
        'action':         'REVIEW',
        'confidence':     'MEDIUM',
        'uncertainty_level': 'MEDIUM',
        'reason':         reason,
        'trigger':        trigger,
    }

def _reject(reason: str, trigger: str) -> Dict[str, Any]:
    return {
        'recommendation': '建议重写',
        'action':         'REJECT',
        'confidence':     'HIGH',
        'uncertainty_level': 'LOW',
        'reason':         reason,
        'trigger':        trigger,
    }