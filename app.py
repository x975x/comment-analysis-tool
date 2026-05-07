"""
评论文本智能分析工具 — Streamlit 主程序
运行方式：streamlit run app.py
"""
import streamlit as st

from lda_core import (
    load_stopwords as lda_load_stopwords,
    load_custom_dict as lda_load_dict,
    preprocess_comments,
    build_corpus,
    compute_coherence_scores,
    train_lda_model,
    get_topics_df,
    get_topic_plotly,
    get_topic_visualization,
)
from cluster_core import (
    load_stopwords as cluster_load_stopwords,
    load_custom_dict as cluster_load_dict,
    load_topic_keywords,
    run_analysis,
)

# ── 页面配置（必须是第一行 st 调用）────────────────────────────
st.set_page_config(
    page_title="评论文本智能分析工具",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全局样式 ─────────────────────────────────────────────────
st.markdown("""
<style>
/* 主标题 */
.main-title {
    font-size: 2rem; font-weight: 700;
    color: #1a73e8; text-align: center;
    padding: 0.5rem 0 1.2rem 0;
}
/* 按钮全宽 */
.stButton > button { width: 100%; border-radius: 8px; font-size: 1rem; }
/* 指标卡 */
div[data-testid="metric-container"] {
    background: #f0f4ff; border-radius: 10px; padding: 0.6rem 1rem;
}
/* 下载按钮行 */
.download-row { margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state 初始化 ─────────────────────────────────────
for key in ('lda_results', 'kw_results'):
    if key not in st.session_state:
        st.session_state[key] = None

# ── 页面标题 ─────────────────────────────────────────────────
st.markdown('<div class="main-title">📊 评论文本智能分析工具</div>', unsafe_allow_html=True)

# ── 侧边栏 ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 选择分析模式")
    mode = st.radio(
        "模式",
        ["📡 模式一：LDA 探索性分析", "🎯 模式二：关键词定向分析"],
        label_visibility="collapsed",
    )
    st.divider()
    if "LDA" in mode:
        st.info(
            "**📡 LDA 探索性分析**\n\n"
            "适合新产品 / 陌生领域。\n"
            "不需要预设主题，算法自动从评论中发现潜在话题，并生成交互式可视化图。"
        )
    else:
        st.info(
            "**🎯 关键词定向分析**\n\n"
            "适合已知产品。\n"
            "按预设主题关键词（如音质、续航、降噪）统计每个维度的提及次数与占比。"
        )
    st.divider()
    st.caption("上传新文件后点击「开始分析」即可刷新结果。")

    # 显示访问地址
    st.divider()
    st.markdown("#### 🌐 共享访问地址")

    # 优先显示 ngrok 公网地址
    ngrok_url = None
    try:
        import urllib.request, json
        with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=1) as r:
            data = json.loads(r.read())
        tunnels = data.get("tunnels", [])
        for t in tunnels:
            if t.get("proto") == "https":
                ngrok_url = t["public_url"]
                break
        if not ngrok_url and tunnels:
            ngrok_url = tunnels[0]["public_url"]
    except Exception:
        pass

    if ngrok_url:
        st.success(f"🚀 **公网地址（任何网络可用）**\n\n{ngrok_url}")
        st.caption("任何人用任何设备打开上方链接即可，无需同一 WiFi。")
    else:
        # 回退显示局域网地址
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            lan_ip = s.getsockname()[0]
            s.close()
        except Exception:
            lan_ip = "未知"
        st.info(f"📡 **局域网地址（同一 WiFi 可用）**\n\nhttp://{lan_ip}:8501")
        st.caption("如需公网访问，请使用「启动工具_公网访问.bat」启动。")


# ════════════════════════════════════════════════════════════
#  模式一：LDA 探索性分析
# ════════════════════════════════════════════════════════════
if "LDA" in mode:
    st.markdown("## 📡 模式一：LDA 探索性主题分析")

    col_up, col_par = st.columns([1.05, 1], gap="large")

    with col_up:
        st.markdown("### 📁 上传文件")
        csv_file  = st.file_uploader("评论数据 *（CSV，必须包含评论列）", type=["csv"], key="lda_csv")
        text_col  = st.text_input("评论所在列名", value="comment", key="lda_col")
        dict_file = st.file_uploader("自定义词典（txt，可选）", type=["txt"], key="lda_dict")
        stop_file = st.file_uploader("停用词文件（txt，可选）", type=["txt"], key="lda_stop")

    with col_par:
        st.markdown("### 🔧 参数设置")
        auto_k = st.checkbox("🔍 自动选择最佳主题数（Coherence Score，较慢）", value=False)
        if auto_k:
            krange = st.slider("主题数搜索范围", min_value=2, max_value=15, value=(2, 8))
        else:
            num_topics = st.slider("主题数量", 2, 20, 5)

        passes    = st.slider("训练轮次 (passes)", 5, 50, 20,
                              help="轮次越高结果越稳定，但训练越慢")
        no_below  = st.slider("词频下限 (no_below)", 1, 20, 3,
                              help="词语至少出现在 N 篇文档中才保留")
        no_above  = st.slider("词频上限 (no_above)", 0.3, 0.95, 0.6, 0.05,
                              help="词语出现在超过 X% 的文档中则过滤")
        num_words = st.slider("每主题展示关键词数", 5, 20, 12)

    st.markdown("")
    run_lda = st.button("🚀 开始 LDA 分析", type="primary", use_container_width=True)

    # ── 运行分析 ────────────────────────────────────────────
    if run_lda:
        if csv_file is None:
            st.error("❌ 请先上传评论 CSV 文件！")
        else:
            with st.status("⏳ 正在分析，请稍候…", expanded=True) as status:
                try:
                    # 1. 读取数据
                    st.write("📂 读取数据…")
                    import pandas as pd
                    df = pd.read_csv(csv_file, encoding='utf-8')
                    if text_col not in df.columns:
                        status.update(label="❌ 列名错误", state="error")
                        st.error(f"找不到列 **'{text_col}'**，CSV 中的列为：{list(df.columns)}")
                        st.stop()
                    st.write(f"✅ 共读取 **{len(df)}** 条评论")

                    # 2. 加载词典
                    st.write("📚 加载停用词 / 自定义词典…")
                    stopwords = lda_load_stopwords(stop_file)
                    if dict_file:
                        lda_load_dict(dict_file)

                    # 3. 分词
                    st.write("✂️ 文本分词与预处理…")
                    docs = preprocess_comments(df, text_col, stopwords)
                    if len(docs) < 10:
                        status.update(label="❌ 文档数不足", state="error")
                        st.error(f"有效文档数仅 {len(docs)} 篇，太少无法分析，请检查数据或停用词。")
                        st.stop()
                    if len(docs) < 50:
                        st.warning(f"⚠️ 有效文档数较少（{len(docs)} 篇），LDA 效果可能不佳。")
                    else:
                        st.write(f"✅ 有效文档：**{len(docs)}** 篇")

                    # 4. 构建语料库
                    st.write("📦 构建词频矩阵…")
                    vectorizer, dtm = build_corpus(docs, no_below, no_above)
                    if dtm.shape[1] == 0:
                        status.update(label="❌ 词典为空", state="error")
                        st.error("过滤后词典为空，请降低「词频下限」或提高「词频上限」。")
                        st.stop()
                    st.write(f"✅ 词典大小：**{dtm.shape[1]}** 个词")

                    # 5. 训练 LDA
                    chart_bytes = None
                    if auto_k:
                        st.write(f"🔍 搜索最佳主题数（{krange[0]} ~ {krange[1]}），耗时约数分钟…")
                        best_k, scores, chart_bytes, lda_model = compute_coherence_scores(
                            dtm, vectorizer, docs, krange[0], krange[1]
                        )
                        final_k = best_k
                        st.write(f"✅ 最佳主题数：**{best_k}**")
                    else:
                        st.write(f"🧠 训练 LDA 模型（{num_topics} 个主题，迭代={passes}）…")
                        lda_model = train_lda_model(dtm, num_topics, passes)
                        final_k = num_topics
                        st.write("✅ 模型训练完成")

                    # 6. 生成可视化
                    st.write("🎨 生成交互式主题可视化…")
                    topics_df       = get_topics_df(lda_model, vectorizer, num_words)
                    topic_fig       = get_topic_plotly(lda_model, dtm, vectorizer, num_words)
                    topic_viz_bytes = get_topic_visualization(lda_model, vectorizer, num_words)

                    # 保存到 session_state
                    st.session_state['lda_results'] = {
                        'topics_df':       topics_df,
                        'topic_fig':       topic_fig,
                        'topic_viz_bytes': topic_viz_bytes,
                        'chart_bytes':     chart_bytes,
                        'final_k':         final_k,
                        'num_docs':        len(docs),
                        'dict_size':       dtm.shape[1],
                    }
                    status.update(label="✅ 分析完成！", state="complete", expanded=False)

                except Exception as e:
                    status.update(label="❌ 分析出错", state="error")
                    st.error(f"错误信息：{e}")
                    st.exception(e)

    # ── 展示结果 ─────────────────────────────────────────────
    if st.session_state['lda_results']:
        res = st.session_state['lda_results']
        st.divider()
        st.markdown("## 📊 分析结果")

        # 指标卡
        m1, m2, m3 = st.columns(3)
        m1.metric("📄 有效文档数", res['num_docs'])
        m2.metric("📖 词典大小",   res['dict_size'])
        m3.metric("🏷️ 主题数量",   res['final_k'])

        # Coherence 折线图（仅 auto_k 时有）
        if res['chart_bytes']:
            st.markdown("### 📈 Perplexity 曲线（越低越好）")
            st.image(res['chart_bytes'], use_column_width=True)

        # 主题关键词表
        st.markdown(f"### 🏷️ 主题关键词（共 {res['final_k']} 个主题）")
        st.dataframe(res['topics_df'], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ 下载主题关键词表（CSV）",
            data=res['topics_df'].to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
            file_name="lda_topics.csv",
            mime="text/csv",
        )

        # 交互式 Plotly 可视化
        st.markdown("### 🎨 LDA 主题交互式可视化")
        st.caption(
            "💡 **左侧气泡图**：每个圆代表一个主题，圆越大表示该主题在语料库中占比越高，"
            "位置由主题相似度决定（越近越相似）。　"
            "**右侧条形图**：显示当前主题的高权重关键词。　"
            "点击上方按钮切换主题，悬停查看精确数值。"
        )
        st.plotly_chart(res['topic_fig'], use_container_width=True)

        # 静态图下载
        with st.expander("⬇️ 下载静态版本（PNG）"):
            st.image(res['topic_viz_bytes'], use_column_width=True)
            st.download_button(
                "⬇️ 下载主题可视化图表（PNG）",
                data=res['topic_viz_bytes'],
                file_name="lda_topic_visualization.png",
                mime="image/png",
            )


# ════════════════════════════════════════════════════════════
#  模式二：关键词定向分析
# ════════════════════════════════════════════════════════════
else:
    st.markdown("## 🎯 模式二：关键词定向主题分析")

    col_up, col_par = st.columns([1.05, 1], gap="large")

    with col_up:
        st.markdown("### 📁 上传文件")
        csv_file   = st.file_uploader("评论数据 *（CSV，必须包含评论列）",             type=["csv"], key="kw_csv")
        text_col   = st.text_input("评论所在列名", value="comment",                     key="kw_col")
        topic_file = st.file_uploader("主题关键词文件 *（txt，格式见右侧说明）",         type=["txt"], key="kw_topic")
        dict_file  = st.file_uploader("自定义词典（txt，可选）",                         type=["txt"], key="kw_dict")
        stop_file  = st.file_uploader("停用词文件（txt，可选）",                         type=["txt"], key="kw_stop")

    with col_par:
        st.markdown("### 🔧 参数设置")
        min_score = st.slider(
            "命中阈值 (min_score)", 1, 5, 1,
            help="一条评论命中该主题至少 N 个关键词才计入此主题",
        )
        st.markdown("#### 📄 主题关键词文件格式")
        st.code(
            "音质 : 音质, 声音, 低音, 高音, 音色\n"
            "降噪 : 降噪, 噪音, 静音, ANC\n"
            "续航 : 续航, 电量, 充电, 电池\n"
            "舒适度 : 舒适, 佩戴, 耳压, 重量\n"
            "连接稳定性 : 蓝牙, 配对, 连接, 延迟, 断连",
            language=None,
        )
        st.caption("每行一个主题，冒号左边是主题名，右边是逗号分隔的关键词。")

    st.markdown("")
    run_kw = st.button("🚀 开始关键词分析", type="primary", use_container_width=True)

    # ── 运行分析 ────────────────────────────────────────────
    if run_kw:
        if csv_file is None:
            st.error("❌ 请先上传评论 CSV 文件！")
        elif topic_file is None:
            st.error("❌ 请上传主题关键词文件！")
        else:
            with st.status("⏳ 正在分析，请稍候…", expanded=True) as status:
                try:
                    import pandas as pd

                    st.write("📂 读取数据…")
                    df = pd.read_csv(csv_file, encoding='utf-8')
                    if text_col not in df.columns:
                        status.update(label="❌ 列名错误", state="error")
                        st.error(f"找不到列 **'{text_col}'**，CSV 中的列为：{list(df.columns)}")
                        st.stop()
                    st.write(f"✅ 共读取 **{len(df)}** 条评论")

                    st.write("📚 加载词典与停用词…")
                    stopwords = cluster_load_stopwords(stop_file)
                    if dict_file:
                        cluster_load_dict(dict_file)
                    topics_dict = load_topic_keywords(topic_file)
                    if not topics_dict:
                        status.update(label="❌ 主题文件解析失败", state="error")
                        st.error("主题关键词文件解析结果为空，请检查格式（主题名 : 词1, 词2）。")
                        st.stop()
                    st.write(f"✅ 已加载 **{len(topics_dict)}** 个主题：{list(topics_dict.keys())}")

                    st.write("🔍 对每条评论进行多标签分类…")
                    result_df, stats_df, bar_bytes, pie_bytes, example_dict, summary = run_analysis(
                        df, text_col, topics_dict, stopwords, min_score
                    )

                    if summary is None:
                        status.update(label="⚠️ 没有命中", state="error")
                        st.error("没有任何评论命中主题，请检查关键词文件或降低命中阈值。")
                        st.stop()

                    st.session_state['kw_results'] = {
                        'result_df':   result_df,
                        'stats_df':    stats_df,
                        'bar_bytes':   bar_bytes,
                        'pie_bytes':   pie_bytes,
                        'example_dict': example_dict,
                        'summary':     summary,
                        'topics_list': list(topics_dict.keys()),
                        'stats_df_lookup': stats_df.set_index('主题'),
                    }
                    status.update(label="✅ 分析完成！", state="complete", expanded=False)

                except Exception as e:
                    status.update(label="❌ 分析出错", state="error")
                    st.error(f"错误信息：{e}")
                    st.exception(e)

    # ── 展示结果 ─────────────────────────────────────────────
    if st.session_state['kw_results']:
        res  = st.session_state['kw_results']
        summ = res['summary']
        lkup = res['stats_df_lookup']

        st.divider()
        st.markdown("## 📊 分析结果")

        # 指标卡
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📄 总评论数",   summ['total'])
        m2.metric("🔢 总提及次数", summ['total_mentions'])
        m3.metric("❓ 未分类评论", summ['unclassified'])
        m4.metric("📉 未分类率",   f"{summ['unclassified_pct']:.1f}%")

        # 统计表
        st.markdown("### 📋 主题提及统计")
        st.dataframe(res['stats_df'], use_container_width=True, hide_index=True)

        # 图表
        st.markdown("### 📈 可视化图表")
        col_bar, col_pie = st.columns(2)
        with col_bar:
            st.markdown("#### 📊 各主题提及次数（条形图）")
            st.image(res['bar_bytes'], use_column_width=True)
        with col_pie:
            st.markdown("#### 🥧 各主题权重分布（饼图）")
            st.image(res['pie_bytes'], use_column_width=True)

        # 示例评论
        st.markdown("### 💬 各主题示例评论")
        for topic in res['topics_list']:
            count    = int(lkup.loc[topic, '提及次数']) if topic in lkup.index else 0
            examples = res['example_dict'].get(topic, [])
            with st.expander(f"【{topic}】— 共提及 {count} 次"):
                if examples:
                    for i, ex in enumerate(examples, 1):
                        st.markdown(f"**{i}.** {ex}")
                else:
                    st.caption("该主题暂无匹配评论")

        # 下载
        st.markdown("### ⬇️ 下载结果")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "📄 下载评论分类结果（CSV）",
                data=res['result_df'].to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
                file_name="classified_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_d2:
            st.download_button(
                "📊 下载主题统计表格（CSV）",
                data=res['stats_df'].to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
                file_name="topic_stats.csv",
                mime="text/csv",
                use_container_width=True,
            )
