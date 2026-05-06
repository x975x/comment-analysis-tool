"""
LDA 分析核心模块
"""
import io
import os
import re
import tempfile

import jieba
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
from gensim import corpora, models
from gensim.models import CoherenceModel

def _set_chinese_font():
    """自动检测可用中文字体，兼容 Windows 和 Linux（Streamlit Cloud）。"""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    candidates = [
        'Microsoft YaHei', 'SimHei',          # Windows
        'Noto Sans CJK SC', 'Noto Sans SC',   # Linux (apt: fonts-noto-cjk)
        'WenQuanYi Micro Hei',                 # Linux 备选
        'DejaVu Sans',                         # 最终兜底（无中文但不报错）
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
])


# ── 停用词 ────────────────────────────────────────────────────
def load_stopwords(file_obj=None):
    """从上传的文件对象加载停用词；若为 None 则使用默认停用词。"""
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
    """将上传的自定义词典写入临时文件并加载到 jieba。"""
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
    """
    对 DataFrame 中指定列的每条评论进行分词和过滤，
    返回由词列表组成的文档列表。
    """
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


# ── 构建语料库 ────────────────────────────────────────────────
def build_corpus(docs, no_below=3, no_above=0.6):
    """
    构建 gensim 词典、词袋语料库和 TF-IDF 语料库。
    - no_below: 词语至少在 N 篇文档中出现才保留
    - no_above: 词语出现在超过 X% 的文档中则过滤
    返回: (dictionary, bow_corpus, tfidf_corpus)
    """
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]
    tfidf = models.TfidfModel(bow_corpus)
    tfidf_corpus = list(tfidf[bow_corpus])
    return dictionary, bow_corpus, tfidf_corpus


# ── 计算 Coherence Score（可选，用于自动选主题数）────────────
def compute_coherence_scores(tfidf_corpus, dictionary, docs, start=2, end=8):
    """
    对 start~end 范围内的每个主题数训练 LDA 并计算 Coherence Score，
    返回最佳主题数、得分列表、折线图字节、最佳模型。
    """
    topic_range = list(range(start, end + 1))
    scores, model_list = [], []

    for num_topics in topic_range:
        lda = models.LdaModel(
            corpus=tfidf_corpus, id2word=dictionary,
            num_topics=num_topics, random_state=42,
            passes=10, alpha='auto', eta='auto',
        )
        model_list.append(lda)
        cm = CoherenceModel(model=lda, texts=docs, dictionary=dictionary, coherence='c_v')
        scores.append(cm.get_coherence())

    best_idx = scores.index(max(scores))
    best_k = topic_range[best_idx]

    # 绘制折线图
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(topic_range, scores, marker='o', color='steelblue', linewidth=2, markersize=8)
    ax.axvline(x=best_k, color='tomato', linestyle='--', alpha=0.8, label=f'最佳 K = {best_k}')
    ax.set_xlabel('主题数量', fontsize=12)
    ax.set_ylabel('Coherence Score', fontsize=12)
    ax.set_title('不同主题数对应的 Coherence Score', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    return best_k, scores, buf.getvalue(), model_list[best_idx]


# ── 训练 LDA 模型 ─────────────────────────────────────────────
def train_lda_model(tfidf_corpus, dictionary, num_topics, passes=20):
    return models.LdaModel(
        corpus=tfidf_corpus, id2word=dictionary,
        num_topics=num_topics, random_state=42,
        passes=passes, alpha='auto', eta='auto',
    )


# ── 提取主题关键词表 ──────────────────────────────────────────
def get_topics_df(lda_model, num_words=12):
    rows = []
    for idx, topic_str in lda_model.print_topics(num_words=num_words):
        keywords = re.findall(r'"([^"]+)"', topic_str)
        rows.append({'主题编号': f'Topic {idx}', '关键词': '，'.join(keywords)})
    return pd.DataFrame(rows)


# ── 生成 pyLDAvis 可视化 HTML ─────────────────────────────────
def get_lda_vis_html(lda_model, bow_corpus, dictionary):
    """
    使用 bow_corpus（原始词频）生成 pyLDAvis 交互可视化，
    返回完整 HTML 字符串。
    """
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dictionary)
    tmp = tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8')
    tmp_name = tmp.name
    tmp.close()
    pyLDAvis.save_html(vis_data, tmp_name)
    with open(tmp_name, 'r', encoding='utf-8') as f:
        html_str = f.read()
    os.unlink(tmp_name)
    return html_str
