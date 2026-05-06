"""
关键词多标签聚类核心模块
"""
import io
import os
import re
import tempfile

import jieba
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def _set_chinese_font():
    """自动检测可用中文字体，兼容 Windows 和 Linux（Streamlit Cloud）。"""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    candidates = [
        'Microsoft YaHei', 'SimHei',
        'Noto Sans CJK SC', 'Noto Sans SC',
        'WenQuanYi Micro Hei',
        'DejaVu Sans',
    ]
    for font in candidates:
        if font in available:
            matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            break
    matplotlib.rcParams['axes.unicode_minus'] = False

_set_chinese_font()

_DEFAULT_STOPWORDS = set([
    '的', '了', '是', '在', '和', '有', '我', '也', '不', '就', '都', '说', '这', '那',
    '他', '她', '它', '我们', '你们', '他们', '这个', '那个', '什么', '怎么', '为什么',
    '感觉', '觉得', '真的', '很', '太', '非常', '有点', '比较', '还行', '不错', '可以',
    '但是', '因为', '所以', '然后', '这样', '那样', '一下', '一直', '一些', '已经',
    '还是', '所有', '大家', '别人', '东西', '产品', '店家', '卖家', '收到', '发货',
    '物流', '快递', '包装', '使用', '体验', '效果', '外观', '材质', '做工', '质量',
    '品牌', '价格', '性价比', '服务', '客服', '态度', '速度', '快', '慢', '好', '差',
    '一般', '满意', '喜欢', '推荐', '购买', '下单', '退货',
])


# ── 停用词 ────────────────────────────────────────────────────
def load_stopwords(file_obj=None):
    if file_obj is None:
        return _DEFAULT_STOPWORDS.copy()
    try:
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return set(line.strip() for line in content.splitlines() if line.strip())
    except Exception:
        return _DEFAULT_STOPWORDS.copy()


# ── 自定义词典 ────────────────────────────────────────────────
def load_custom_dict(file_obj):
    if file_obj is None:
        return
    tmp = tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False)
    tmp.write(file_obj.read())
    tmp.close()
    try:
        jieba.load_userdict(tmp.name)
    finally:
        os.unlink(tmp.name)


# ── 主题关键词文件解析 ────────────────────────────────────────
def load_topic_keywords(file_obj):
    """
    解析主题关键词文件，格式：
        主题名称 : 关键词1, 关键词2, 关键词3
    返回 dict: {主题名: [关键词列表]}
    """
    topics = {}
    content = file_obj.read()
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    for line in content.splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        topic, kw_str = line.split(':', 1)
        topic = topic.strip()
        keywords = [kw.strip() for kw in kw_str.split(',') if kw.strip()]
        if topic and keywords:
            topics[topic] = keywords
    return topics


# ── 分词 ─────────────────────────────────────────────────────
def _segment(comment, stopwords):
    text = re.sub(r'[^一-龥a-zA-Z0-9]', ' ', str(comment))
    words = jieba.lcut(text)
    return set(w.strip() for w in words if len(w.strip()) >= 2 and w.strip() not in stopwords)


# ── 主分析函数 ────────────────────────────────────────────────
def run_analysis(df, text_col, topics_dict, stopwords, min_score=1):
    """
    对每条评论进行多标签分类，返回：
    (result_df, stats_df, bar_bytes, pie_bytes, example_dict, summary)
    若没有任何命中，summary 返回 None。
    """
    total = len(df)
    records = []

    for _, row in df.iterrows():
        comment = str(row[text_col])
        words = _segment(comment, stopwords)
        hit = [
            t for t, kws in topics_dict.items()
            if sum(1 for k in kws if k in words) >= min_score
        ]
        records.append({
            '原始评论': comment,
            '命中主题': '，'.join(hit) if hit else '未分类',
            '_hits': hit,
        })

    # 统计
    topic_count = {t: sum(1 for r in records if t in r['_hits']) for t in topics_dict}
    unclassified = sum(1 for r in records if not r['_hits'])
    total_mentions = sum(topic_count.values())

    if total_mentions == 0:
        return None, None, None, None, None, None

    mention_rate = {t: c / total * 100 for t, c in topic_count.items()}
    weight = {t: c / total_mentions * 100 for t, c in topic_count.items()}

    # 统计表
    stats_df = pd.DataFrame([{
        '主题': t,
        '提及次数': topic_count[t],
        '提及率(%)': round(mention_rate[t], 1),
        '权重(%)': round(weight[t], 1),
    } for t in topics_dict])

    # 分类结果表
    result_df = pd.DataFrame([{
        '原始评论': r['原始评论'],
        '命中主题': r['命中主题'],
    } for r in records])

    topics_list = list(topics_dict.keys())

    # ── 条形图 ───────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(max(10, len(topics_list) * 1.3), 5))
    counts = [topic_count[t] for t in topics_list]
    bars = ax1.bar(topics_list, counts, color='steelblue', edgecolor='white', linewidth=0.8)
    ax1.set_xlabel('主题', fontsize=12)
    ax1.set_ylabel('提及次数（评论数）', fontsize=12)
    ax1.set_title('各主题被提及次数（多标签，一条评论可计入多个主题）', fontsize=13)
    ax1.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            str(val), ha='center', va='bottom', fontsize=11,
        )
    plt.tight_layout()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # ── 饼图 ─────────────────────────────────────────────────
    weights_list = [weight[t] for t in topics_list]
    colors = list(plt.cm.Set3.colors)
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.pie(
        weights_list, labels=topics_list, autopct='%1.1f%%', startangle=90,
        colors=[colors[i % len(colors)] for i in range(len(topics_list))],
    )
    ax2.set_title('各主题权重分布（基于提及次数归一化）', fontsize=13)
    ax2.axis('equal')
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # 示例评论（每个主题最多3条）
    example_dict = {
        t: [r['原始评论'] for r in records if t in r['_hits']][:3]
        for t in topics_dict
    }

    summary = {
        'total': total,
        'total_mentions': total_mentions,
        'unclassified': unclassified,
        'unclassified_pct': unclassified / total * 100,
    }

    return result_df, stats_df, buf1.getvalue(), buf2.getvalue(), example_dict, summary
