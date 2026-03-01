"""
批量审核主程序 - 支持单文件或文件夹拖拽
"""

import sys
import os
import json
import numpy as np
import configparser

# 路径设置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# 读取配置
config = configparser.ConfigParser()
config.read(os.path.join(PROJECT_ROOT, 'config', 'config.ini'), encoding='utf-8')

WEIGHTS = np.array([
    config.getfloat('weights', 'fact'),
    config.getfloat('weights', 'brand'),
    config.getfloat('weights', 'compliance'),
    config.getfloat('weights', 'norm')
])

MC_CONFIG = {
    'n_simulations': config.getint('monte_carlo', 'n_simulations'),
    'perturbation_sigma': config.getfloat('monte_carlo', 'perturbation_sigma'),
    'random_seed': config.getint('monte_carlo', 'random_seed'),
    'base_weights': WEIGHTS
}

PATHS = {
    'facts_db': config.get('paths', 'facts_db'),
    'synonyms_db': config.get('paths', 'synonyms_db'),
    'output_dir': config.get('paths', 'output_dir')
}

# 导入模块
from fact_checker import FactChecker
from semantic_checker import SemanticChecker
from monte_carlo import MonteCarloAuditor
from visualizer import plot_audit_distribution, plot_multi_case_comparison


def audit_single_file(file_path, fact_checker, semantic_checker, mc_auditor):
    """审核单个文件"""
    print(f"\n审核文件: {os.path.basename(file_path)}")
    
    # 读取文本
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 执行审核流程
    fact_score, fact_errors = fact_checker.check(text)
    semantic_scores, semantic_matches = semantic_checker.check(text)
    mc_result = mc_auditor.audit(fact_score, semantic_scores[:3])
    
    # 生成图表
    case_name = os.path.splitext(os.path.basename(file_path))[0]
    vis_path = plot_audit_distribution(
        scores=mc_result['scores'],
        case_name=case_name,
        decision_info={
            'recommendation': mc_result['decision']['recommendation'],
            'pass_probability': mc_result['pass_probability']
        },
        output_path=os.path.join(PATHS['output_dir'], f"{case_name}_audit.png")
    )
    
    # 返回结果
    return {
        'file': file_path,
        'case_name': case_name,
        'text_preview': text[:100] + '...' if len(text) > 100 else text,
        'fact_score': fact_score,
        'semantic_scores': semantic_scores.tolist(),
        'monte_carlo': {
            'mean': float(mc_result['mean']),
            'median': float(mc_result['median']),
            'pass_probability': float(mc_result['pass_probability']),
            'decision': mc_result['decision']['recommendation'],
            'action': mc_result['decision']['action']
        },
        'chart': vis_path
    }


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python batch_audit.py <文件或文件夹路径>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # 初始化模块
    print("=" * 60)
    print("图书馆AIGC内容审核工具")
    print("=" * 60)
    print("正在初始化...")
    
    fact_checker = FactChecker(PATHS['facts_db'])
    semantic_checker = SemanticChecker(PATHS['synonyms_db'])
    mc_auditor = MonteCarloAuditor(MC_CONFIG)
    
    os.makedirs(PATHS['output_dir'], exist_ok=True)
    
    results = []
    
    # 判断是文件还是文件夹
    if os.path.isfile(input_path):
        # 单个文件
        if input_path.endswith('.txt'):
            result = audit_single_file(input_path, fact_checker, semantic_checker, mc_auditor)
            results.append(result)
        else:
            print(f"跳过非TXT文件: {input_path}")
    
    elif os.path.isdir(input_path):
        # 文件夹：处理所有TXT文件
        txt_files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
        print(f"找到 {len(txt_files)} 个TXT文件")
        
        for i, filename in enumerate(txt_files, 1):
            print(f"\n[{i}/{len(txt_files)}] ", end='')
            file_path = os.path.join(input_path, filename)
            result = audit_single_file(file_path, fact_checker, semantic_checker, mc_auditor)
            results.append(result)
        
        # 生成多案例对比图
        if len(results) > 1:
            print("\n生成多案例对比图...")
            comparison_path = plot_multi_case_comparison(
                results,
                os.path.join(PATHS['output_dir'], 'multi_case_comparison.png')
            )
            print(f"对比图: {comparison_path}")
    
    else:
        print(f"错误: 路径不存在 - {input_path}")
        sys.exit(1)
    
    # 保存汇总报告
    summary = {
        'total': len(results),
        'cases': results
    }
    
    with open(os.path.join(PATHS['output_dir'], 'audit_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("审核汇总")
    print("=" * 60)
    for r in results:
        print(f"{r['case_name']}: {r['monte_carlo']['decision']} "
              f"(通过概率: {r['monte_carlo']['pass_probability']:.1%})")
    
    print(f"\n结果保存至: {os.path.abspath(PATHS['output_dir'])}")


if __name__ == '__main__':
    main()