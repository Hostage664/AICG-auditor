"""
visualizer.py
AIGC 审核结果可视化模块
将 batch_audit 传入的 monte_carlo 字段渲染为 HTML 报告。
图表部分使用纯 CSS 实现（无 matplotlib 依赖），展示：
  - 决策摘要
  - 三态概率横向条形图（CSS 渲染）
  - 得分分布数轴（均值/中位数/置信区间标注）
  - 完整 JSON 折叠展示
"""

import json
import logging
from datetime import datetime
from pathlib  import Path
from typing   import Any, Dict

logger = logging.getLogger(__name__)

_ACTION_COLOR = {"APPROVE": "#27ae60", "REVIEW": "#f39c12", "REJECT": "#e74c3c"}
_ACTION_LABEL = {"APPROVE": "通过",    "REVIEW": "复审",    "REJECT": "拒绝"}

# ══════════════════════════════════════════════════════════════════════════════
class AuditVisualizer:
    """
    将蒙特卡洛审核结果转换为独立 HTML 报告（纯 CSS 图表，零外部依赖）。

    Parameters
    ----------
    output_dir : str | Path
        HTML 文件输出目录，不存在时自动创建。
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Visualizer] 初始化完成，输出目录: {self.output_dir}")

    # ── 公开接口 ────────────────────────────────────────────────────────────────

    def generate_report(
        self,
        audit_result: Dict[str, Any],
        filename:     str,
    ) -> str:
        """
        生成 HTML 报告并写入磁盘。

        Parameters
        ----------
        audit_result : dict
            batch_audit 输出的完整审核结果（含 'monte_carlo' 键）。
        filename : str
            不含扩展名的基础文件名。

        Returns
        -------
        str  生成的 HTML 文件绝对路径。
        """
        mc       = audit_result.get("monte_carlo", {})
        mc_flat  = _parse_mc(mc)
        html     = self._build_html(mc_flat, audit_result, filename)

        out_path = self.output_dir / f"{filename}_report.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info(f"[Visualizer] 报告已写入: {out_path}")
        return str(out_path)

    # ── HTML 组装 ───────────────────────────────────────────────────────────────

    def _build_html(
        self,
        mc_flat:     Dict[str, Any],
        full_result: Dict[str, Any],
        filename:    str,
    ) -> str:
        """组装完整 HTML 字符串（纯 CSS 图表）。"""

        action     = mc_flat["action"]
        decision   = mc_flat["decision_text"]
        color      = _ACTION_COLOR.get(action, "#7f8c8d")
        label      = _ACTION_LABEL.get(action, action)

        mean    = mc_flat["mean"]
        median  = mc_flat["median"]
        std     = mc_flat["std"]
        ci_lo   = mc_flat["ci_lo"]
        ci_hi   = mc_flat["ci_hi"]
        trigger = mc_flat["trigger"]
        reason  = mc_flat["reason"]

        pass_p  = mc_flat["pass_probability"]   * 100
        rev_p   = mc_flat["review_probability"] * 100
        rej_p   = mc_flat["reject_probability"] * 100

        pass_thr   = mc_flat["pass_threshold"]    # 0.82
        review_thr = mc_flat["review_threshold"]  # 0.60

        # ── 概率条 ────────────────────────────────────────────────────────────
        prob_bars = (
            _prob_bar("通过", pass_p, "#27ae60")
            + _prob_bar("复审", rev_p, "#f39c12")
            + _prob_bar("拒绝", rej_p, "#e74c3c")
        )

        # ── 得分数轴 CSS 图 ───────────────────────────────────────────────────
        score_axis = _score_axis(
            mean, median, std, ci_lo, ci_hi,
            pass_thr, review_thr,
        )

        # ── 各维度得分 ────────────────────────────────────────────────────────
        scores     = full_result.get("scores", {})
        score_rows = "".join(
            _score_row(k, v)
            for k, v in scores.items()
        )

        raw_json = json.dumps(full_result, ensure_ascii=False, indent=2)

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>审核报告 · {filename}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
      background: #f0f2f5; color: #2c3e50;
      padding: 28px 20px; font-size: 14px;
    }}
    .wrap {{ max-width: 900px; margin: 0 auto; }}

    /* ── 标题 ── */
    .page-title {{
      font-size: 1.45em; font-weight: 700;
      color: #1a252f; margin-bottom: 22px;
    }}
    .page-title em {{ color: #3498db; font-style: normal; }}

    /* ── 卡片 ── */
    .card {{
      background: #fff; border-radius: 10px;
      padding: 18px 22px; margin-bottom: 18px;
      box-shadow: 0 2px 8px rgba(0,0,0,.07);
    }}
    .card-title {{
      font-size: .95em; font-weight: 600; color: #34495e;
      margin-bottom: 14px; padding-bottom: 8px;
      border-bottom: 2px solid #ecf0f1;
    }}

    /* ── 决策徽章 ── */
    .badge {{
      display: inline-block; padding: 4px 14px;
      border-radius: 20px; font-weight: 700;
      font-size: 1em; color: #fff; letter-spacing: .5px;
    }}

    /* ── 摘要 grid ── */
    .sum-grid {{
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 14px 32px;
    }}
    .sum-row {{ display: flex; flex-direction: column; gap: 3px; }}
    .sum-key {{ font-size: .78em; color: #888; }}
    .sum-val {{ font-size: 1em; font-weight: 600; }}
    .sum-full {{ grid-column: 1 / -1; }}

    /* ── 概率条 ── */
    .prob-row {{
      display: flex; align-items: center; gap: 10px; margin: 7px 0;
    }}
    .prob-label {{ width: 32px; color: #555; font-size: .9em; }}
    .prob-track {{
      flex: 1; height: 14px; background: #ecf0f1;
      border-radius: 7px; overflow: hidden; position: relative;
    }}
    .prob-fill {{
      height: 100%; border-radius: 7px;
      transition: width .5s ease;
    }}
    .prob-pct {{
      width: 52px; text-align: right;
      font-size: .9em; font-weight: 700;
    }}

    /* ── 得分数轴 ── */
    .axis-wrap {{
      position: relative; height: 64px;
      margin: 18px 0 8px;
    }}
    /* 背景分区色块 */
    .axis-bg {{
      position: absolute; top: 24px; height: 16px;
      border-radius: 3px;
    }}
    /* 数轴基线 */
    .axis-line {{
      position: absolute; top: 32px; left: 0; right: 0;
      height: 2px; background: #ccc;
    }}
    /* 标记针 */
    .axis-pin {{
      position: absolute; top: 14px;
      transform: translateX(-50%);
      display: flex; flex-direction: column; align-items: center;
      gap: 2px;
    }}
    .axis-pin .pin-dot {{
      width: 10px; height: 10px;
      border-radius: 50%; border: 2px solid #fff;
      box-shadow: 0 0 0 2px currentColor;
    }}
    .axis-pin .pin-label {{
      font-size: .7em; font-weight: 600;
      white-space: nowrap; margin-top: 2px;
    }}
    /* CI 区间线 */
    .axis-ci {{
      position: absolute; top: 30px; height: 6px;
      background: rgba(52,152,219,.35);
      border-radius: 3px;
    }}
    /* 阈值竖线 */
    .axis-thr {{
      position: absolute; top: 18px; width: 2px; height: 28px;
      transform: translateX(-50%);
    }}
    .axis-thr-label {{
      position: absolute; top: 0;
      font-size: .68em; font-weight: 600;
      transform: translateX(-50%);
      white-space: nowrap;
    }}
    /* 刻度标签 */
    .axis-tick {{
      position: absolute; top: 50px;
      font-size: .7em; color: #888;
      transform: translateX(-50%);
    }}

    /* ── 各维度得分 ── */
    .score-table {{ width: 100%; border-collapse: collapse; }}
    .score-table td {{ padding: 6px 4px; vertical-align: middle; }}
    .score-table td:first-child {{
      width: 120px; color: #777; font-size: .88em;
    }}
    .score-table td:last-child {{
      width: 52px; text-align: right;
      font-weight: 700; font-size: .9em;
    }}
    .score-bar-track {{
      height: 10px; background: #ecf0f1;
      border-radius: 5px; overflow: hidden;
    }}
    .score-bar-fill {{
      height: 100%; border-radius: 5px;
    }}

    /* ── JSON 折叠 ── */
    details summary {{
      cursor: pointer; color: #3498db;
      font-size: .88em; user-select: none; padding: 4px 0;
    }}
    details summary:hover {{ text-decoration: underline; }}
    .json-pre {{
      background: #1e1e2e; color: #cdd6f4;
      border-radius: 6px; padding: 14px;
      font-size: .76em; line-height: 1.6;
      overflow: auto; max-height: 540px;
      margin-top: 10px; white-space: pre;
      font-family: "Cascadia Code","Fira Code","Consolas",monospace;
    }}

    /* ── footer ── */
    .footer {{
      text-align: center; color: #bbb;
      font-size: .78em; margin-top: 6px;
    }}
  </style>
</head>
<body>
<div class="wrap">

  <h1 class="page-title">📊 AIGC 内容审核报告 · <em>{filename}</em></h1>

  <!-- ① 决策摘要 -->
  <div class="card">
    <div class="card-title">🎯 蒙特卡洛决策摘要</div>
    <div class="sum-grid">

      <div class="sum-row">
        <span class="sum-key">最终处置</span>
        <span class="sum-val">
          <span class="badge" style="background:{color};">{label}</span>
          &nbsp;<span style="color:#888;font-weight:400;font-size:.9em;">{decision}</span>
        </span>
      </div>

      <div class="sum-row">
        <span class="sum-key">均值 / 中位数</span>
        <span class="sum-val">{mean:.4f} / {median:.4f}</span>
      </div>

      <div class="sum-row">
        <span class="sum-key">标准差</span>
        <span class="sum-val">{std:.4f}</span>
      </div>

      <div class="sum-row">
        <span class="sum-key">95% 置信区间</span>
        <span class="sum-val">[{ci_lo:.4f}, {ci_hi:.4f}]</span>
      </div>

      <div class="sum-row sum-full">
        <span class="sum-key">触发规则</span>
        <span class="sum-val">{trigger}</span>
      </div>

      <div class="sum-row sum-full">
        <span class="sum-key">原因说明</span>
        <span class="sum-val" style="font-weight:400;color:#555;">{reason}</span>
      </div>

    </div>
  </div>

  <!-- ② 概率分布 -->
  <div class="card">
    <div class="card-title">🎲 三态概率分布</div>
    {prob_bars}
  </div>

  <!-- ③ 得分数轴图（核心可视化） -->
  <div class="card">
    <div class="card-title">📈 得分分布数轴</div>
    {score_axis}
  </div>

  <!-- ④ 各维度得分 -->
  <div class="card">
    <div class="card-title">📋 各维度得分</div>
    <table class="score-table">
      {score_rows}
    </table>
  </div>

  <!-- ⑤ 原始 JSON -->
  <div class="card">
    <div class="card-title">📄 完整审核数据（JSON）</div>
    <details>
      <summary>▶ 展开 / 收起原始 JSON</summary>
      <pre class="json-pre">{raw_json}</pre>
    </details>
  </div>

  <div class="footer">
    生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    &nbsp;·&nbsp; library-aigc-auditor
  </div>

</div>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _parse_mc(mc: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 batch_audit 传入的扁平化 monte_carlo 字典解析为 visualizer 内部参数包。

    batch_audit 实际传入字段：
        mean, median, std, ci_95,
        pass_probability, review_probability, reject_probability,
        decision(str), action(str), reason(str), trigger(str)
    """
    ci = mc.get("ci_95", [0.0, 1.0])
    ci_lo = float(ci[0]) if len(ci) >= 2 else 0.0
    ci_hi = float(ci[1]) if len(ci) >= 2 else 1.0

    return {
        "action":        str(mc.get("action",   "UNKNOWN")),
        "decision_text": str(mc.get("decision", "—")),
        "reason":        str(mc.get("reason",   "—")),
        "trigger":       str(mc.get("trigger",  "—")),

        "mean":   float(mc.get("mean",   0.0)),
        "median": float(mc.get("median", 0.0)),
        "std":    float(mc.get("std",    0.0)),
        "ci_lo":  ci_lo,
        "ci_hi":  ci_hi,

        "pass_probability":   float(mc.get("pass_probability",   0.0)),
        "review_probability": float(mc.get("review_probability", 0.0)),
        "reject_probability": float(mc.get("reject_probability", 0.0)),

        # batch_audit 未传时使用 mc_config 默认值
        "pass_threshold":   float(mc.get("pass_threshold",   0.82)),
        "review_threshold": float(mc.get("review_threshold", 0.60)),
    }

def _prob_bar(label: str, pct: float, color: str) -> str:
    """渲染单条概率进度条 HTML（纯 CSS）。"""
    return f"""
    <div class="prob-row">
      <span class="prob-label">{label}</span>
      <div class="prob-track">
        <div class="prob-fill" style="width:{pct:.1f}%;background:{color};"></div>
      </div>
      <span class="prob-pct" style="color:{color};">{pct:.1f}%</span>
    </div>"""

def _pct(val: float) -> str:
    """将 [0,1] 得分值转为百分比字符串，用于 CSS left 定位。"""
    v = max(0.0, min(1.0, val))
    return f"{v * 100:.2f}%"

def _score_axis(
    mean:       float,
    median:     float,
    std:        float,
    ci_lo:      float,
    ci_hi:      float,
    pass_thr:   float,
    review_thr: float,
) -> str:
    """
    用纯 CSS 绝对定位绘制得分数轴：
      - 背景分区色块（红/橙/绿）
      - 95% CI 区间蓝色色块
      - 通过/复审阈值竖线
      - 均值（蓝点）与中位数（紫点）标记针
      - 0 / 0.5 / 1.0 刻度
    """
    # ── 背景分区 ───────────────────────────────────────────────────────────────
    bg_reject  = (
        f'<div class="axis-bg" '
        f'style="left:0;width:{review_thr*100:.2f}%;'
        f'background:rgba(231,76,60,.15);"></div>'
    )
    bg_review  = (
        f'<div class="axis-bg" '
        f'style="left:{review_thr*100:.2f}%;'
        f'width:{(pass_thr - review_thr)*100:.2f}%;'
        f'background:rgba(243,156,18,.15);"></div>'
    )
    bg_pass    = (
        f'<div class="axis-bg" '
        f'style="left:{pass_thr*100:.2f}%;'
        f'width:{(1.0 - pass_thr)*100:.2f}%;'
        f'background:rgba(39,174,96,.15);"></div>'
    )

    # ── 95% CI 色块 ────────────────────────────────────────────────────────────
    ci_width = max(0.0, ci_hi - ci_lo)
    ci_block = (
        f'<div class="axis-ci" '
        f'style="left:{_pct(ci_lo)};width:{ci_width*100:.2f}%;"></div>'
    )

    # ── 阈值竖线 ───────────────────────────────────────────────────────────────
    thr_pass = (
        f'<div class="axis-thr" '
        f'style="left:{_pct(pass_thr)};background:#27ae60;"></div>'
        f'<span class="axis-thr-label" '
        f'style="left:{_pct(pass_thr)};color:#27ae60;top:-14px;">'
        f'通过 {pass_thr}</span>'
    )
    thr_review = (
        f'<div class="axis-thr" '
        f'style="left:{_pct(review_thr)};background:#f39c12;"></div>'
        f'<span class="axis-thr-label" '
        f'style="left:{_pct(review_thr)};color:#f39c12;top:-14px;">'
        f'复审 {review_thr}</span>'
    )

    # ── 均值标记针 ─────────────────────────────────────────────────────────────
    pin_mean = (
        f'<div class="axis-pin" style="left:{_pct(mean)};color:#2980b9;">'
        f'  <div class="pin-dot" style="background:#2980b9;"></div>'
        f'  <span class="pin-label">均值<br>{mean:.3f}</span>'
        f'</div>'
    )

    # ── 中位数标记针 ───────────────────────────────────────────────────────────
    pin_median = (
        f'<div class="axis-pin" style="left:{_pct(median)};color:#8e44ad;">'
        f'  <div class="pin-dot" style="background:#8e44ad;"></div>'
        f'  <span class="pin-label">中位数<br>{median:.3f}</span>'
        f'</div>'
    )

    # ── 刻度标签 ───────────────────────────────────────────────────────────────
    ticks = "".join(
        f'<span class="axis-tick" style="left:{v*100:.0f}%;">{v:.1f}</span>'
        for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    # ── σ 注释 ─────────────────────────────────────────────────────────────────
    sigma_note = (
        f'<div style="margin-top:44px;font-size:.78em;color:#888;">'
        f'σ = {std:.4f} &nbsp;·&nbsp; '
        f'95% CI [{ci_lo:.4f}, {ci_hi:.4f}]'
        f'</div>'
    )

    return f"""
    <div style="position:relative;">
      <!-- 图例 -->
      <div style="display:flex;gap:16px;font-size:.78em;color:#666;margin-bottom:6px;">
        <span><span style="display:inline-block;width:10px;height:10px;
              background:#2980b9;border-radius:50%;margin-right:4px;"></span>均值</span>
        <span><span style="display:inline-block;width:10px;height:10px;
              background:#8e44ad;border-radius:50%;margin-right:4px;"></span>中位数</span>
        <span><span style="display:inline-block;width:10px;height:10px;
              background:rgba(52,152,219,.4);margin-right:4px;"></span>95% CI</span>
        <span><span style="display:inline-block;width:2px;height:10px;
              background:#27ae60;margin-right:4px;"></span>通过阈值</span>
        <span><span style="display:inline-block;width:2px;height:10px;
              background:#f39c12;margin-right:4px;"></span>复审阈值</span>
      </div>
      <div class="axis-wrap">
        {bg_reject}{bg_review}{bg_pass}
        <div class="axis-line"></div>
        {ci_block}
        {thr_pass}{thr_review}
        {pin_mean}{pin_median}
        {ticks}
      </div>
      {sigma_note}
    </div>"""

def _score_row(key: str, val: Any) -> str:
    """渲染单行维度得分（名称 + 进度条 + 数值）。"""
    _label_map = {
        "fact_score":       "事实核查",
        "brand_score":      "品牌合规",
        "compliance_score": "规定合规",
        "norm_score":       "规范表达",
    }
    label = _label_map.get(key, key)
    v     = max(0.0, min(1.0, float(val)))
    color = (
        "#27ae60" if v >= 0.75 else
        "#f39c12" if v >= 0.45 else
        "#e74c3c"
    )
    return f"""
    <tr>
      <td>{label}</td>
      <td>
        <div class="score-bar-track">
          <div class="score-bar-fill"
               style="width:{v*100:.1f}%;background:{color};"></div>
        </div>
      </td>
      <td style="color:{color};">{v:.3f}</td>
    </tr>"""