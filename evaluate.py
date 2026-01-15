"""
Agent评估脚本

功能：
- 合并agent_results.csv和ListFinalnew.xlsx（ground truth）
- 使用GPT-4o对agent的final_decision进行结构化评分
- 评分维度：治疗方案一致性、临床推理质量、安全性考量、指南参考准确性、完整性
- 生成统计报告和可视化图表

使用方法：
python evaluate.py [--agent-csv PATH] [--list-final PATH] [--output-dir DIR]
"""

import os
import json
import asyncio
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
from typing import Optional, Dict, Any

# 配置管理器
from config_manager import ConfigManager

# OpenAI客户端
from openai import AsyncOpenAI

# 可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    warnings.warn("matplotlib未安装，将跳过图表生成。安装: pip install matplotlib")

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    warnings.warn("seaborn未安装，将使用matplotlib绘图。安装: pip install seaborn")

# 统计检验库
try:
    from scipy import stats
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    warnings.warn("scipy未安装，将跳过统计检验。安装: pip install scipy")


# ============================================================================
# 配置
# ============================================================================

# 初始化配置管理器
config_manager = ConfigManager()

# 从config.toml加载openai_gpt4o配置
gpt4o_config = config_manager.get_config("openai_gpt4o")

# OpenAI配置
API_BASE = gpt4o_config.get("api_base")
API_KEY = gpt4o_config.get("api_key")
OPENAI_MODEL = gpt4o_config.get("model_name", "gpt-4o")
MAX_TOKENS = gpt4o_config.get("max_tokens", 1500)
TIMEOUT = gpt4o_config.get("timeout", 300)
TEST_LIMIT = int(os.getenv("TEST_LIMIT", "0"))

# 验证API配置
if not API_BASE or not API_KEY or API_KEY == "EMPTY":
    print("⚠️  警告: OpenAI API配置未找到！")
    print("请设置环境变量:")
    print("  - OPENAI_API_BASE: OpenAI API基础URL")
    print("  - OPENAI_API_KEY: OpenAI API密钥")
    print("\n或在.env文件中配置这些变量。")
    sys.exit(1)

# 评估维度（10分制）
DIMENSIONS = [
    "treatment_match",
    "clinical_reasoning",
    "safety_awareness",
    "guideline_compliance",
    "completeness"
]

DIMENSION_NAMES = {
    "treatment_match": "Treatment Match",
    "clinical_reasoning": "Clinical Reasoning",
    "safety_awareness": "Safety Awareness",
    "guideline_compliance": "Guideline Compliance",
    "completeness": "Completeness",
    "overall": "Overall (Weighted Avg)"
}

# 维度权重（用于计算overall加权平均）
DIMENSION_WEIGHTS = {
    "treatment_match": 0.40,        # 核心：与ground truth对比
    "clinical_reasoning": 0.20,     # 独立评估：推理质量
    "safety_awareness": 0.15,       # 独立评估：安全性考量
    "guideline_compliance": 0.10,   # 独立评估：指南引用
    "completeness": 0.15            # 独立评估：完整性
}

# 10分制范围
SCORE_MIN = 1
SCORE_MAX = 10

# 使用配置管理器的异步客户端
client = config_manager.async_external_client

# 如果需要使用openai_gpt4o配置，创建专用客户端
# 注意：config_manager默认使用local_gpt配置，这里我们需要openai_gpt4o
try:
    # 创建使用openai_gpt4o配置的专用客户端
    gpt4o_client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
        timeout=TIMEOUT
    )
    client = gpt4o_client
except Exception as e:
    print(f"❌ 初始化OpenAI客户端失败: {e}")
    print("请检查环境变量OPENAI_API_BASE和OPENAI_API_KEY是否正确设置。")
    sys.exit(1)


# ============================================================================
# 1. 合并功能
# ============================================================================
def merge_agent_and_ground_truth(agent_csv: str, list_final_xlsx: str) -> pd.DataFrame:
    """
    合并agent_results.csv和ListFinalnew.xlsx
    
    参数:
        agent_csv: agent结果CSV路径
        list_final_xlsx: ground truth Excel路径
    
    返回:
        合并后的DataFrame
    """
    print("=" * 80)
    print("📊 开始合并数据...")
    print("=" * 80)
    
    # 读取agent_results.csv
    print(f"\n[1/2] 读取 {agent_csv}...")
    agent_path = Path(agent_csv)
    if not agent_path.exists():
        print(f"   ✗ 文件不存在: {agent_csv}")
        sys.exit(1)
    
    df_agent = pd.read_csv(agent_path, encoding='utf-8-sig')
    print(f"   ✓ Agent结果: {len(df_agent)} 条记录")
    print(f"   列名: {list(df_agent.columns)}")
    
    # 读取ListFinalnew.xlsx
    print(f"\n[2/2] 读取 {list_final_xlsx}...")
    list_path = Path(list_final_xlsx)
    if not list_path.exists():
        print(f"   ✗ 文件不存在: {list_final_xlsx}")
        sys.exit(1)
    
    df_list = pd.read_excel(list_path)
    print(f"   ✓ Ground Truth: {len(df_list)} 条记录")
    print(f"   列名: {list(df_list.columns)}")
    
    # 过滤掉不需要的列（如final_decision_old）
    # 只保留patient_id, timestamp和ground_truth相关列
    keep_columns = ['patient_id', 'timestamp']
    # 保留所有ground_truth开头的列
    keep_columns.extend([col for col in df_list.columns if col.startswith('ground_truth')])
    df_list = df_list[keep_columns]
    print(f"   保留列: {list(df_list.columns)}")
    
    # 统一数据类型
    df_agent['patient_id'] = df_agent['patient_id'].astype(str)
    df_agent['timestamp'] = df_agent['timestamp'].astype(str)
    df_list['patient_id'] = df_list['patient_id'].astype(str)
    df_list['timestamp'] = df_list['timestamp'].astype(str)
    
    # 合并
    print("\n" + "=" * 80)
    print("合并数据...")
    print("=" * 80)
    
    merged_df = pd.merge(
        df_agent,
        df_list,
        on=['patient_id', 'timestamp'],
        how='outer',
        suffixes=('_agent', '_list')
    )
    
    print(f"   ✓ 合并后记录数: {len(merged_df)}")
    
    # 排序
    merged_df = merged_df.sort_values(['patient_id', 'timestamp'], ascending=[True, True])
    merged_df = merged_df.reset_index(drop=True)
    
    # 统计信息
    print("\n" + "=" * 80)
    print("合并完成！")
    print("=" * 80)
    print(f"总记录数: {len(merged_df)}")
    print(f"唯一患者数: {merged_df['patient_id'].nunique()}")
    print(f"列数: {len(merged_df.columns)}")
    
    return merged_df


# ============================================================================
# 2. 评估功能
# ============================================================================
async def evaluate_decision(
    decision: str,
    ground_truth: str,
    max_tokens: int = MAX_TOKENS
) -> Optional[Dict[str, Any]]:
    """使用GPT-4o评估决策（10分制）"""
    prompt = f"""You are an expert hematological oncologist evaluating clinical decision quality.

GROUND TRUTH (Brief clinical decision summary):
{ground_truth}

DECISION TO EVALUATE:
{decision}

Score each dimension on a 1-10 scale (1=worst, 10=best):

{{
  "treatment_match": {{"score": <1-10>, "reason": "<brief reason>"}},
  "clinical_reasoning": {{"score": <1-10>, "reason": "<brief reason>"}},
  "safety_awareness": {{"score": <1-10>, "reason": "<brief reason>"}},
  "guideline_compliance": {{"score": <1-10>, "reason": "<brief reason>"}},
  "completeness": {{"score": <1-10>, "reason": "<brief reason>"}}
}}

DIMENSION DEFINITIONS:

=== A. COMPARATIVE EVALUATION (vs Ground Truth) ===

1. treatment_match (治疗方案匹配度) [Weight: 40%]:
   Does the DECISION recommend the SAME core treatment as GROUND TRUTH?
   - Same drugs/regimen = high score
   - Similar approach but different drugs = medium score  
   - Different treatment strategy = low score

=== B. INDEPENDENT EVALUATION (Decision Quality) ===

2. clinical_reasoning (临床推理质量) [Weight: 20%]:
   Is the clinical reasoning in the DECISION sound and evidence-based?
   Consider: disease staging, risk stratification, patient factors, logical coherence.

3. safety_awareness (安全性考量) [Weight: 15%]:
   Does the DECISION adequately address safety considerations?
   Consider: contraindications, dose adjustments for organ dysfunction, drug interactions, monitoring needs.

4. guideline_compliance (指南引用质量) [Weight: 10%]:
   How specific and accurate are the guideline references in the DECISION?
   - Specific citations (version, section, evidence level) = high score
   - General references with some detail = medium score
   - Only generic mentions like "per NCCN/ESMO" = low score (≤4)
   - No references = very low score

5. completeness (完整性) [Weight: 15%]:
   How comprehensive is the DECISION?
   Consider coverage of: diagnosis, treatment plan, supportive care, follow-up, patient-specific considerations.

Be objective. Use the full 1-10 scale. Output ONLY valid JSON."""
    
    try:
        response = await client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            instructions="You are an expert hematological oncologist. Always respond with valid JSON only."
        )
        
        content = response.output_text.strip()
        
        # 去除可能的markdown代码块标记
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        result = json.loads(content)
        return result
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        return None
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        return None


def calculate_overall_score(eval_result: Dict[str, Any]) -> float:
    """计算五个维度的加权平均分作为overall score"""
    if not eval_result:
        return None
    
    total_weight = 0
    weighted_sum = 0
    
    for dim, weight in DIMENSION_WEIGHTS.items():
        if dim in eval_result and eval_result[dim].get("score") is not None:
            score = eval_result[dim]["score"]
            weighted_sum += score * weight
            total_weight += weight
    
    if total_weight > 0:
        return round(weighted_sum / total_weight, 2)
    return None


def flatten_evaluation(eval_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """将评估结果扁平化为CSV列格式，并计算overall加权平均"""
    if not eval_result:
        result = {}
        for dim in DIMENSIONS:
            result[f"{dim}_score"] = None
            result[f"{dim}_reason"] = "Evaluation failed"
        result["overall_score"] = None
        return result
    
    result = {}
    for dimension, values in eval_result.items():
        result[f"{dimension}_score"] = values.get("score")
        result[f"{dimension}_reason"] = values.get("reason", "")
    
    # 计算加权平均作为overall score
    overall = calculate_overall_score(eval_result)
    result["overall_score"] = overall
    
    return result


async def process_row(row_id: int, row_data: pd.Series, total_rows: int, ground_truth_col: str) -> Optional[Dict[str, Any]]:
    """处理单行数据，评估agent的决策"""
    print(f"\n{'='*80}")
    print(f"📊 处理行 {row_id}/{total_rows}")
    
    ground_truth = row_data.get(ground_truth_col, "")
    decision = row_data.get("final_decision", "")
    
    # 检查有效性
    if pd.isna(ground_truth) or not str(ground_truth).strip():
        print(f"⚠️  行 {row_id}: Ground truth为空，跳过")
        return None
    
    if pd.isna(decision) or not str(decision).strip():
        print(f"⚠️  行 {row_id}: Agent决策为空，跳过")
        return None
    
    result = {
        "Row_ID": row_id,
        "patient_id": row_data.get("patient_id", ""),
        "timestamp": row_data.get("timestamp", ""),
        "final_decision": decision,
        "ground_truth": ground_truth
    }
    
    # 评估决策
    print(f"🔄 评估Agent决策...")
    eval_result = await evaluate_decision(str(decision), str(ground_truth))
    scores = flatten_evaluation(eval_result)
    result.update(scores)
    
    if eval_result:
        overall = calculate_overall_score(eval_result)
        print(f"✅ Agent Overall: {overall}/10")
    
    return result


async def run_evaluation(merged_df: pd.DataFrame, ground_truth_col: str) -> pd.DataFrame:
    """运行评估流程"""
    total_rows = len(merged_df)
    print(f"\n📊 开始评估，共 {total_rows} 行数据")
    
    # 应用测试限制
    if TEST_LIMIT > 0:
        merged_df = merged_df.head(TEST_LIMIT)
        print(f"⚠️  测试模式: 仅处理前 {TEST_LIMIT} 行")
    
    print(f"🤖 使用模型: {OPENAI_MODEL}")
    print("="*80)
    
    # 处理每一行
    results = []
    for idx, row in merged_df.iterrows():
        row_id = idx + 1
        result = await process_row(row_id, row, len(merged_df), ground_truth_col)
        if result:
            results.append(result)
        await asyncio.sleep(0.5)  # 避免API限流
    
    if not results:
        print("\n❌ 没有成功处理的数据")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print(f"✅ 评估完成!")
    print(f"📊 成功处理: {len(results)} 行")
    
    return result_df


# ============================================================================
# 3. 分析功能
# ============================================================================
def analyze_scores(df: pd.DataFrame):
    """分析评分数据"""
    print("\n" + "="*80)
    print("📊 评分统计分析")
    print("="*80)
    
    print(f"\n【Agent 评分统计】")
    print("-" * 80)
    
    # 五个维度
    for dim in DIMENSIONS:
        col = f"{dim}_score"
        if col in df.columns:
            scores = df[col].dropna()
            if len(scores) > 0:
                mean_score = scores.mean()
                weight_str = f"(权重{DIMENSION_WEIGHTS[dim]:.0%})" if dim in DIMENSION_WEIGHTS else ""
                print(f"  {DIMENSION_NAMES[dim]:20s}: {mean_score:5.2f}/10  "
                      f"(最小: {scores.min():.1f}, 最大: {scores.max():.1f}, n={len(scores)}) {weight_str}")
    
    # Overall加权平均
    overall_col = "overall_score"
    if overall_col in df.columns:
        overall_scores = df[overall_col].dropna()
        if len(overall_scores) > 0:
            mean_overall = overall_scores.mean()
            print(f"  {'─'*60}")
            print(f"  {DIMENSION_NAMES['overall']:20s}: {mean_overall:5.2f}/10  "
                  f"(最小: {overall_scores.min():.1f}, 最大: {overall_scores.max():.1f})")


def score_distribution(df: pd.DataFrame):
    """评分分布统计（10分制）"""
    print("\n" + "="*80)
    print("📊 Overall评分分布")
    print("="*80)
    
    # 10分制分布区间
    score_ranges = [(1, 3), (4, 6), (7, 8), (9, 10)]
    range_labels = ["低(1-3)", "中(4-6)", "良好(7-8)", "优秀(9-10)"]
    
    col = "overall_score"
    if col not in df.columns:
        return
    
    scores = df[col].dropna()
    if len(scores) == 0:
        return
    
    print(f"\n【Agent Overall评分分布】")
    print("-" * 40)
    for (low, high), label in zip(score_ranges, range_labels):
        count = ((scores >= low) & (scores <= high)).sum()
        percentage = (count / len(scores)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {label:12s}: {count:3d} ({percentage:5.1f}%) {bar}")


# ============================================================================
# 4. 可视化功能
# ============================================================================
def create_boxplots(df: pd.DataFrame, output_dir: Path):
    """生成箱线图"""
    if not PLOT_AVAILABLE:
        print("\n⚠️  matplotlib未安装，跳过图表生成")
        return
    
    print("\n" + "="*80)
    print("📊 生成箱线图")
    print("="*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义颜色
    color = '#3498db'  # 蓝色
    
    # 1. 所有维度 + Overall 的综合箱线图 (2x3 = 6个子图)
    all_dims = DIMENSIONS + ["overall"]  # 5个维度 + 1个overall
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle('Agent Performance Across All Dimensions', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, dim in enumerate(all_dims):
        # 确定列名
        col = "overall_score" if dim == "overall" else f"{dim}_score"
        
        if col in df.columns:
            scores = df[col].dropna()
            if len(scores) > 0:
                if SEABORN_AVAILABLE:
                    sns.boxplot(y=scores, ax=axes[idx], color=color)
                    sns.swarmplot(y=scores, ax=axes[idx], color='black', alpha=0.2, size=2)
                else:
                    bp = axes[idx].boxplot([scores], patch_artist=True)
                    bp['boxes'][0].set_facecolor(color)
                    bp['boxes'][0].set_alpha(0.7)
                
                title = DIMENSION_NAMES.get(dim, dim)
                # Overall用不同颜色背景突出显示
                if dim == "overall":
                    axes[idx].set_facecolor('#f0f0f0')
                    title = "★ " + title
                axes[idx].set_title(title, fontweight='bold')
                axes[idx].set_ylabel('Score (1-10)')
                axes[idx].set_ylim(0, 11)
                axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "boxplot_all_dimensions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {output_file}")
    plt.close()
    
    # 2. Overall加权平均评分对比图（单独大图）
    fig, ax = plt.subplots(figsize=(8, 7))
    
    col = "overall_score"
    if col in df.columns:
        scores = df[col].dropna()
        if len(scores) > 0:
            if SEABORN_AVAILABLE:
                sns.boxplot(y=scores, ax=ax, color=color, width=0.4)
                sns.swarmplot(y=scores, ax=ax, color='black', alpha=0.2, size=3)
            else:
                bp = ax.boxplot([scores], patch_artist=True)
                bp['boxes'][0].set_facecolor(color)
                bp['boxes'][0].set_alpha(0.7)
            
            ax.set_title('Agent Overall Score', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score (1-10)', fontsize=12)
            ax.set_ylim(0, 11)
            ax.grid(True, alpha=0.3, axis='y')
            
            stats_text = f"μ={scores.mean():.2f}, σ={scores.std():.2f}, n={len(scores)}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / "boxplot_overall.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {output_file}")
    plt.close()
    
    # 3. 雷达图（各维度平均分）
    create_radar_chart(df, output_dir)


def create_radar_chart(df: pd.DataFrame, output_dir: Path):
    """生成雷达图"""
    if not PLOT_AVAILABLE:
        return
    
    print("\n📊 生成雷达图")
    
    # 计算各维度的平均分
    means = []
    for dim in DIMENSIONS:
        col = f"{dim}_score"
        if col in df.columns:
            score = df[col].dropna().mean()
            means.append(score if not np.isnan(score) else 0)
        else:
            means.append(0)
    
    # 雷达图
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    color = '#3498db'
    
    values = means + means[:1]  # 闭合
    ax.plot(angles, values, 'o-', linewidth=2, label='Agent', color=color)
    ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DIMENSION_NAMES[d] for d in DIMENSIONS], fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_title('Agent Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    output_file = output_dir / "radar_chart.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {output_file}")
    plt.close()


def run_analysis(df: pd.DataFrame, output_dir: Path):
    """运行分析流程"""
    print("\n" + "="*80)
    print("📊 开始分析评估结果")
    print("="*80)
    
    # 文本统计分析
    analyze_scores(df)
    score_distribution(df)
    
    # 生成可视化图表
    if PLOT_AVAILABLE:
        create_boxplots(df, output_dir)
        print(f"\n📁 所有图表已保存到: {output_dir}")


# ============================================================================
# 主函数
# ============================================================================
async def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Agent评估脚本 - 合并、评估、分析Agent决策",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluate.py
  python evaluate.py --agent-csv path/to/agent_results.csv
  python evaluate.py --agent-csv agent.csv --list-final ground_truth.xlsx --output-dir my_eval
        """
    )
    
    parser.add_argument(
        '--agent-csv',
        type=str,
        default='agent_results.csv',
        help='Agent结果CSV文件路径 (默认: agent_results.csv)'
    )
    
    parser.add_argument(
        '--list-final',
        type=str,
        default='agent_eval/ground_truth/ListFinalnew.xlsx',
        help='Ground Truth Excel文件路径 (默认: agent_eval/ground_truth/ListFinalnew.xlsx)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='agent_eval/results',
        help='输出目录 (默认: agent_eval/results)'
    )
    
    parser.add_argument(
        '--ground-truth-column',
        type=str,
        default='ground_truth_eng',
        help='Ground truth列名 (默认: ground_truth_eng)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🏥 Agent临床决策评估工具")
    print("="*80)
    print(f"📄 Agent CSV: {args.agent_csv}")
    print(f"📄 Ground Truth: {args.list_final}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"📊 评估维度: {', '.join(DIMENSION_NAMES.values())}")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 步骤1: 合并数据
    merged_df = merge_agent_and_ground_truth(args.agent_csv, args.list_final)
    
    # 保存合并结果
    merged_output = output_dir / "merged_agent_results.csv"
    merged_df.to_csv(merged_output, index=False, encoding='utf-8-sig')
    print(f"\n✓ 合并结果已保存: {merged_output}")
    
    # 步骤2: 检查是否已有评估结果
    eval_output = output_dir / "evaluation_results.csv"
    
    if eval_output.exists():
        print(f"\n发现已有评估结果: {eval_output}")
        user_input = input("是否重新评估? (y/n, 默认n跳过评估直接分析): ").strip().lower()
        
        if user_input == 'y':
            eval_df = await run_evaluation(merged_df, args.ground_truth_column)
            if not eval_df.empty:
                eval_df.to_csv(eval_output, index=False, encoding="utf-8-sig")
                print(f"💾 评估结果已保存: {eval_output}")
        else:
            print("跳过评估，直接加载已有结果进行分析...")
            eval_df = pd.read_csv(eval_output, encoding="utf-8")
    else:
        # 运行评估
        eval_df = await run_evaluation(merged_df, args.ground_truth_column)
        if not eval_df.empty:
            eval_df.to_csv(eval_output, index=False, encoding="utf-8-sig")
            print(f"💾 评估结果已保存: {eval_output}")
    
    if eval_df.empty:
        print("❌ 没有数据可供分析")
        return
    
    # 步骤3: 运行分析
    # figures目录在agent_eval下，与results平级
    figures_dir = output_dir.parent / "figures" if output_dir.name == "results" else output_dir / "figures"
    run_analysis(eval_df, figures_dir)
    
    print("\n" + "="*80)
    print("✅ 所有任务完成!")
    print("="*80)
    print(f"📁 输出目录: {output_dir}")
    print(f"   - merged_agent_results.csv (合并数据)")
    print(f"   - evaluation_results.csv (评估结果)")
    print(f"   - figures/ (可视化图表)")


if __name__ == "__main__":
    asyncio.run(main())
