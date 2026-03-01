"""
可视化模块 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional
import os


def setup_matplotlib_style():
    """设置matplotlib"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 24,
        'font.weight': 'bold',
    })


def plot_audit_distribution(
    scores: np.ndarray,
    case_name: str,
    decision_info: Dict,
    output_path: Optional[str] = None
) -> str:
    """
    绘制单个案例的审核置信度分布
    """
    setup_matplotlib_style()
    
    mean = float(scores.mean())
    median = float(np.median(scores))
    std = float(scores.std())
    
    pass_threshold = 0.8
    review_threshold = 0.6
    
    fig = plt.figure(figsize=(20, 16), dpi=150)
    fig.suptitle(
        f'{case_name} - 审核置信度分析\n'
        f'决策: {decision_info["recommendation"]} | '
        f'通过概率: {decision_info["pass_probability"]:.1%}',
        fontsize=22, fontweight='bold', y=0.98
    )
    
    # 频率分布
    ax1 = plt.subplot(2, 1, 1)
    counts, bin_edges = np.histogram(scores, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ax1.plot(bin_centers, counts, color='darkblue', linewidth=3.0, 
             alpha=0.95, label='置信度分布')
    ax1.fill_between(bin_centers, counts, alpha=0.3, color='lightblue')
    
    ax1.axvline(x=median, color='red', linestyle='--', linewidth=2,
                label=f'中位数: {median:.3f}')
    ax1.axvline(x=mean, color='green', linestyle=':', linewidth=2,
                label=f'均值: {mean:.3f}')
    
    ax1.axvspan(pass_threshold, 1.0, alpha=0.2, color='green', label='通过区域')
    ax1.axvspan(review_threshold, pass_threshold, alpha=0.2, color='orange', label='复核区域')
    ax1.axvspan(0, review_threshold, alpha=0.2, color='red', label='拒绝区域')
    
    ax1.set_xlabel('审核通过概率', fontsize=18, fontweight='bold')
    ax1.set_ylabel('概率密度', fontsize=18, fontweight='bold')
    ax1.set_title('置信度频率分布', fontsize=20, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=14, framealpha=0.95)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim([0, 1])
    
    # 累积频率
    ax2 = plt.subplot(2, 1, 2)
    sorted_scores = np.sort(scores)
    cumulative_freq = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    
    ax2.plot(sorted_scores, cumulative_freq, color='darkgreen', 
             linewidth=3.0, alpha=0.95, label='累积频率')
    ax2.fill_between(sorted_scores, cumulative_freq, alpha=0.3, color='lightgreen')
    
    ax2.set_xlabel('审核通过概率', fontsize=18, fontweight='bold')
    ax2.set_ylabel('累积频率', fontsize=18, fontweight='bold')
    ax2.set_title('累积频率分布', fontsize=20, fontweight='bold', pad=15)
    ax2.legend(loc='lower right', fontsize=14, framealpha=0.95)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    if output_path is None:
        output_path = f'{case_name}_audit_distribution.png'
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_multi_case_comparison(
    results: List[Dict],
    output_path: str = 'multi_case_comparison.png'
) -> str:
    """
    多案例对比图
    """
    setup_matplotlib_style()
    
    n_cases = len(results)
    n_cols = min(4, n_cases)
    n_rows = (n_cases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), dpi=150)
    if n_cases == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_cases > 1 else [axes]
    
    fig.suptitle('多案例审核置信度对比分析', fontsize=26, fontweight='bold', y=0.98)
    
    for idx, result in enumerate(results):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        scores = result['monte_carlo']['scores']
        case_name = result['case_name']
        decision = result['final_decision']['recommendation']
        
        counts, bin_edges = np.histogram(scores, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        color_map = {
            '通过': 'green',
            '谨慎通过': 'lightgreen',
            '人工复核': 'orange',
            '重审': 'red',
            '建议重审': 'darkred'
        }
        color = color_map.get(decision, 'blue')
        
        ax.plot(bin_centers, counts, color=color, linewidth=2.5)
        ax.fill_between(bin_centers, counts, alpha=0.3, color=color)
        
        median = float(np.median(scores))
        ax.axvline(x=median, color='black', linestyle='--', linewidth=1.5)
        
        pass_prob = result['monte_carlo']['pass_probability']
        ax.set_title(
            f'{case_name}\n{decision} | 通过:{pass_prob:.0%} | 中位:{median:.2f}',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlim([0, 1])
        ax.set_xlabel('通过概率', fontsize=10)
        ax.set_ylabel('密度', fontsize=10)
        ax.grid(True, alpha=0.2)
    
    # 隐藏多余的子图
    for idx in range(n_cases, len(axes)):
        if idx < len(axes):
            axes[idx].axis('off')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path