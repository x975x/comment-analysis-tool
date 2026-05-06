"""
LDA 分析核心模块
使用 scikit-learn 实现，兼容 Windows 本地运行和 Streamlit Cloud 部署。
"""
import io
import os
import re
import tempfile

import jieba
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.sklearn as pyLDAvis_sk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# ── 中文字体自动适配（Windows / Linux 均可用）────────────────
def _set_chinese_font():
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    candidates = [
        'Microsoft YaHei', 'SimHei',          # Windows
        'Noto Sans CJK SC', 'Noto Sans SC',   # Linux (apt: fonts-noto-cjk)
        'WenQuanYi Micro Hei',                 # Linux 备选
        'DejaVu Sans',                         # 兜底（无中文不报错）
    ]
    for font in candidates:
        if font in available:
            matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            break
    matplotlib.rcParams['axes.unicode_minus'] = False

_set_chinese_font()


# ── 停用词 ────────────────────────────────────────────────────
_DEFAULT_STOPWORDS = set([
    '的', '了', '是', '在', '和', '有', '我', '也', '不', '就', '都', '说', '这', '那',
    '他', '她', '它', '我们', '你们', '他们', '这个', '那个', '什么', '怎么', '为什么',
])

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


# ── 文本预处理 ────────────────────────────────────────────────
def preprocess_comments(df, text_col, stopwords):
    docs = []
    for comment in df[text_col].dropna():
        if not isinstance(comment, str):
            continue
        text = re.sub(r'[^一-龥a-zA-Z0-9]', ' ', comment)
        words = [
            w.strip() for w in jieba.lcut(text)
            if len(w.strip()) >= 2 and w.strip() not in stopwords
        ]
        if words:
            docs.append(words)
    return docs


# ── 构建词频矩阵（替代 gensim 语料库）────────────────────────
def build_corpus(docs, no_below=3, no_above=0.6):
    """
    用 sklearn CountVectorizer 构建文档-词频矩阵。
    - no_below: 词语至少在 N 篇文档中出现才保留（绝对数量）
    - no_above: 词语出现比例超过此值则过滤（0~1）
    返回: (vectorizer, dtm)
    """
    texts = [' '.join(doc) for doc in docs]
    vectorizer = CountVectorizer(
        min_df=no_below,
        max_df=no_above,
        token_pattern=r'(?u)\b\w\w+\b',   # 匹配2字符以上的词
    )
    dtm = vectorizer.fit_transform(texts)
    return vectorizer, dtm


# ── 自动选主题数（用 Perplexity，越低越好）───────────────────
def compute_coherence_scores(dtm, vectorizer, docs, start=2, end=8):
    """
    对 start~end 范围内的每个 K 训练 LDA，用 Perplexity 选最优。
    返回: (best_k, perplexity列表, 折线图bytes, 最优模型)
    """
    topic_range = list(range(start, end + 1))
    perplexities, model_list = [], []

    for k in topic_range:
        lda = LatentDirichletAllocation(
            n_components=k, random_state=42,
            max_iter=10, learning_method='online',
        )
        lda.fit(dtm)
        model_list.append(lda)
        perplexities.append(lda.perplexity(dtm))

    best_idx = int(np.argmin(perplexities))
    best_k   = topic_range[best_idx]

    # 绘制折线图
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(topic_range, perplexities, marker='o', color='steelblue', linewidth=2, markersize=8)
    ax.axvline(x=best_k, color='tomato', linestyle='--', alpha=0.8, label=f'最佳 K = {best_k}')
    ax.set_xlabel('主题数量', fontsize=12)
    ax.set_ylabel('Perplexity（越低越好）', fontsize=12)
    ax.set_title('不同主题数对应的 Perplexity', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    return best_k, perplexities, buf.getvalue(), model_list[best_idx]


# ── 训练 LDA 模型 ─────────────────────────────────────────────
def train_lda_model(dtm, num_topics, passes=20):
    lda = LatentDirichletAllocation(
        n_components=num_topics, random_state=42,
        max_iter=passes, learning_method='online',
    )
    lda.fit(dtm)
    return lda


# ── 提取主题关键词表 ──────────────────────────────────────────
def get_topics_df(lda_model, vectorizer, num_words=12):
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for idx, topic_vec in enumerate(lda_model.components_):
        top_idx  = topic_vec.argsort()[:-num_words - 1:-1]
        keywords = [feature_names[i] for i in top_idx]
        rows.append({'主题编号': f'Topic {idx}', '关键词': '，'.join(keywords)})
    return pd.DataFrame(rows)


# ── 生成 pyLDAvis 可视化 HTML ─────────────────────────────────
def get_lda_vis_html(lda_model, dtm, vectorizer):
    vis_data = pyLDAvis_sk.prepare(lda_model, dtm, vectorizer)
    tmp = tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8')
    tmp_name = tmp.name
    tmp.close()
    pyLDAvis.save_html(vis_data, tmp_name)
    with open(tmp_name, 'r', encoding='utf-8') as f:
        html_str = f.read()
    os.unlink(tmp_name)
    return html_str
