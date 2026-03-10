"""
batch_audit.py
批量审核主程序
支持：单文件 / 文件夹 两种输入模式
依赖模块：fact_checker / semantic_checker / monte_carlo / visualizer
用法：
    python src/batch_audit.py <文件路径.txt>
    python src/batch_audit.py <文件夹路径>
"""

import sys
import json
import logging
import configparser
import traceback
from datetime import datetime
from pathlib  import Path

import numpy as np

# ── 路径初始化（兼容直接运行与启动器调用）────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))

import os
os.chdir(_PROJECT_ROOT)

from fact_checker     import FactChecker
from semantic_checker import SemanticChecker
from monte_carlo      import MonteCarloAuditor
from visualizer       import AuditVisualizer

# ── 日志初始化 ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers = [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── 配置加载 ──────────────────────────────────────────────────────────────────
def _load_config() -> dict:
    """从 config.ini 读取所有配置，返回 {'paths': ..., 'mc_config': ...}"""
    cfg      = configparser.ConfigParser()
    cfg_path = _PROJECT_ROOT / 'config' / 'config.ini'

    if not cfg_path.exists():
        logger.warning(f"[BatchAudit] 未找到 config.ini，使用全部默认值: {cfg_path}")
    else:
        cfg.read(str(cfg_path), encoding='utf-8')

    paths = {
        'facts_db':     _PROJECT_ROOT / cfg.get('paths', 'facts_db',     fallback='config/facts_db.json'),
        'blacklist_db': _PROJECT_ROOT / cfg.get('paths', 'blacklist_db', fallback='config/blacklist_db.json'),
        'whitelist_db': _PROJECT_ROOT / cfg.get('paths', 'whitelist_db', fallback='config/whitelist_db.json'),
        'output_dir':   _PROJECT_ROOT / cfg.get('paths', 'output_dir',  fallback='output'),
        'log_file':     _PROJECT_ROOT / cfg.get('paths', 'log_file',    fallback='output/audit.log'),
    }

    mc_config = {
        'base_weights': [
            cfg.getfloat('monte_carlo', 'weight_fact',       fallback=0.50),
            cfg.getfloat('monte_carlo', 'weight_brand',      fallback=0.18),
            cfg.getfloat('monte_carlo', 'weight_compliance', fallback=0.20),
            cfg.getfloat('monte_carlo', 'weight_norm',       fallback=0.12),
        ],
        'perturbation_sigma': cfg.getfloat('monte_carlo', 'perturbation_sigma', fallback=0.05),
        'weight_bounds': (
            cfg.getfloat('monte_carlo', 'weight_bound_min', fallback=0.05),
            cfg.getfloat('monte_carlo', 'weight_bound_max', fallback=0.75),
        ),
        'n_simulations':    cfg.getint  ('monte_carlo', 'n_simulations',    fallback=10_000),
        'pass_threshold':   cfg.getfloat('monte_carlo', 'pass_threshold',   fallback=0.82),
        'review_threshold': cfg.getfloat('monte_carlo', 'review_threshold', fallback=0.60),
        'random_seed':      cfg.getint  ('monte_carlo', 'random_seed',      fallback=42),
        # 硬性否决阈值
        'single_dim_reject': cfg.getfloat('hard_reject', 'single_dim_reject_threshold', fallback=0.45),
        'bc_avg_reject':     cfg.getfloat('hard_reject', 'brand_compliance_avg_reject', fallback=0.50),
        # 硬性复核阈值
        'fact_review_floor': cfg.getfloat('hard_review', 'fact_score_review_threshold', fallback=0.95),
        'pass_prob_min':     cfg.getfloat('hard_review', 'pass_prob_min_for_approve',   fallback=0.60),
    }

    return {'paths': paths, 'mc_config': mc_config}

# ── 单文件审核 ────────────────────────────────────────────────────────────────
def _audit_single(
    text:             str,
    filename:         str,
    fact_checker:     FactChecker,
    semantic_checker: SemanticChecker,
    mc_auditor:       MonteCarloAuditor,
) -> dict:
    """对单份文本执行完整审核流程，返回结构化结果"""

    # ① 事实核查
    fact_score, fact_issues = fact_checker.check(text)
    logger.info(
        f"[BatchAudit] 事实得分={fact_score:.3f}  问题数={len(fact_issues)}"
    )

    # ② 语义检测
    sem_scores, sem_matches = semantic_checker.check(text)
    brand_score      = float(sem_scores[0])
    compliance_score = float(sem_scores[1])
    norm_score       = float(sem_scores[2])
    logger.info(
        f"[BatchAudit] 语义得分: "
        f"品牌={brand_score:.3f}  合规={compliance_score:.3f}  规范={norm_score:.3f}"
    )

# ③ 蒙特卡洛综合评分 + 决策
    mc_result = mc_auditor.audit(fact_score, sem_scores)

    return {
        'file': filename,
        'scores': {
            'fact_score':       fact_score,
            'brand_score':      brand_score,
            'compliance_score': compliance_score,
            'norm_score':       norm_score,
        },
        'fact_issues': fact_issues,
        'sem_matches': sem_matches,
        'monte_carlo': {
            # ── 统计量 ────────────────────────────────────────────────────
            'mean':               float(mc_result['mean']),
            'median':             float(mc_result.get('median', 0.0)),
            'std':                float(mc_result.get('std',    0.0)),
            'ci_95':              mc_result.get('ci_95', [0.0, 1.0]),
            'n_simulations':      int(mc_result.get('n_simulations', 0)),  # ★ 补齐

            # ── 阈值（visualizer 绘图需要）────────────────────────────────
            'pass_threshold':     float(mc_result.get('pass_threshold',   0.82)),  # ★ 补齐
            'review_threshold':   float(mc_result.get('review_threshold', 0.60)),  # ★ 补齐

            # ── 概率 ──────────────────────────────────────────────────────
            'pass_probability':   float(mc_result['pass_probability']),
            'review_probability': float(mc_result.get('review_probability', 0.0)),
            'reject_probability': float(mc_result.get('reject_probability', 0.0)),

            # ── 决策──────────────────────────────────────
            'decision':           mc_result['decision']['recommendation'],
            'action':             mc_result['decision']['action'],
            'reason':             mc_result['decision'].get('reason',  ''),
            'trigger':            mc_result['decision'].get('trigger', ''),

            # ── 分布数据────────────────────────────
            'score_distribution': mc_result.get('score_distribution', {}),
        },
    }

# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("用法: python batch_audit.py <文件或文件夹路径>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    print('=' * 65)
    print('  AIGC 内容审核工具')
    print('=' * 65)
    logger.info(f"[BatchAudit] 输入路径: {input_path.resolve()}")

    # ── 加载配置 ──────────────────────────────────────────────────────────────
    config    = _load_config()
    paths     = config['paths']
    mc_config = config['mc_config']

    # 确保输出目录存在
    paths['output_dir'].mkdir(parents=True, exist_ok=True)

    # ── 模块初始化 ────────────────────────────────────────────────────────────
    logger.info("[BatchAudit] 正在初始化各模块...")
    try:
        fact_checker = FactChecker(str(paths['facts_db']))

        semantic_checker = SemanticChecker(
            blacklist_db_path = str(paths['blacklist_db']),
            whitelist_db_path = str(paths['whitelist_db']),
        )

        mc_auditor = MonteCarloAuditor(mc_config)

        # 接入 visualizer，输出目录与 JSON 报告共用同一目录
        visualizer = AuditVisualizer(output_dir=str(paths['output_dir']))

    except Exception as e:
        logger.error(f"[BatchAudit] 模块初始化失败: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    logger.info("[BatchAudit] 所有模块初始化完成")

    # ── 收集待审核文件 ────────────────────────────────────────────────────────
    if input_path.is_file():
        if input_path.suffix.lower() != '.txt':
            logger.error(f"[BatchAudit] 仅支持 .txt 文件，收到: {input_path.suffix}")
            sys.exit(1)
        txt_files = [input_path]

    elif input_path.is_dir():
        txt_files = sorted(input_path.glob('*.txt'))
        if not txt_files:
            logger.warning(f"[BatchAudit] 目录中未找到 .txt 文件: {input_path}")
            sys.exit(0)

    else:
        logger.error(f"[BatchAudit] 路径不存在或类型不支持: {input_path}")
        sys.exit(1)

    logger.info(f"[BatchAudit] 共 {len(txt_files)} 个文件待审核")

    # ── 批量审核 ──────────────────────────────────────────────────────────────
    results   = []
    html_reports = []   # 收集所有 HTML 报告路径，最终汇总时写入 summary

    for idx, txt_path in enumerate(txt_files, 1):
        logger.info(f"[BatchAudit] [{idx}/{len(txt_files)}] 审核: {txt_path.name}")
        try:
            text   = txt_path.read_text(encoding='utf-8')
            result = _audit_single(
                text             = text,
                filename         = txt_path.name,
                fact_checker     = fact_checker,
                semantic_checker = semantic_checker,
                mc_auditor       = mc_auditor,
            )
            results.append(result)

            # ── 写出单文件 summary JSON ────────────────────────────────────
            out_json = paths['output_dir'] / f"{txt_path.stem}_audit_summary.json"
            out_json.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )

            # ── 生成可视化 HTML 报告 ───────────────────────────────────────
            try:
                html_path = visualizer.generate_report(
                    audit_result = result,
                    filename     = txt_path.stem,
                )
                html_reports.append(html_path)
                logger.info(f"[BatchAudit] 可视化报告: {html_path}")
            except Exception as viz_err:
                # visualizer 失败不影响主流程，降级为警告
                logger.warning(
                    f"[BatchAudit] 可视化生成失败（不影响审核结果）: {viz_err}\n"
                    f"{traceback.format_exc()}"
                )

            logger.info(
                f"[BatchAudit] 完成: {txt_path.name}  "
                f"→ {result['monte_carlo']['action']}  "
                f"({result['monte_carlo']['decision']})"
            )

        except Exception as e:
            logger.error(
                f"[BatchAudit] 审核流程异常: {txt_path.stem}\n"
                f"{traceback.format_exc()}"
            )

    # ── 汇总报告 ──────────────────────────────────────────────────────────────
    summary_path = paths['output_dir'] / 'batch_summary.json'
    summary_path.write_text(
        json.dumps(
            {
                'generated_at': datetime.now().isoformat(),
                'total':        len(txt_files),
                'audited':      len(results),
                'html_reports': html_reports,   # 记录所有 HTML 报告路径
                'results':      results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )

    logger.info(f"[BatchAudit] 汇总报告: {summary_path}")
    print('=' * 65)
    print(f"  审核完成，共处理 {len(results)}/{len(txt_files)} 个文件")
    print(f"  结果目录: {paths['output_dir'].resolve()}")
    if html_reports:
        print(f"  HTML报告: {len(html_reports)} 份")
    print('=' * 65)

if __name__ == '__main__':
    main()