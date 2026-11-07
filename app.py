# -*- coding: utf-8 -*-
"""查找小助手：面向软件测试论文的证据型阅读助手。"""

import os
import streamlit as st
import streamlit.components.v1 as components

from core import (
    extract_pages, chunk_pages, search, ask_deepseek,
    translate_text, write_discussion, extract_references,
)
from graph_utils import build_citation_svg


st.set_page_config(
    page_title="查找小助手",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1150px;}
    h1 {color: #0B3D33;}
    .small-note {color: #6F7C76; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def default_key():
    try:
        secret_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    except Exception:
        secret_key = ""
    return str(secret_key or os.getenv("DEEPSEEK_API_KEY", ""))


def clear_all():
    for k in ("chunks", "pages", "stats", "qa_result", "trans_result",
              "graph_refs", "write_result"):
        st.session_state.pop(k, None)


def render_qa(api_key, chunks, top_k, compare):
    st.subheader("论文问答")
    st.caption("只根据上传的论文回答；每句结论都要标 [文件名 第X页]；没证据就拒答。")

    with st.expander("可以问什么？点开看示例"):
        st.markdown(
            """
- 这篇论文在评估时用了哪些 Java 项目？给出页码。
- 这几篇论文分别用了什么数据集和指标？做一个对比表。
- 这篇论文的威胁效度（threats to validity）是什么？
- 这篇论文有没有公开代码和数据集？链接在哪？
- 这篇论文的工业落地客户是谁？（预期：拒答，论文里没写）
            """
        )

    question = st.text_input(
        "问论文里的问题",
        placeholder="例如：List the Java projects used in the evaluation. Give the page number.",
        key="qa_question",
    )
    ask = st.button("提问", type="primary", key="qa_ask")

    if ask:
        if not question.strip():
            st.warning("先写一个问题。")
        elif not api_key.strip():
            st.warning("左侧先填 DeepSeek API Key。")
        else:
            k = max(top_k, 10) if compare else top_k
            hits, top_score = search(question, chunks, top_k=k)
            if not hits or top_score < 0.08:
                st.session_state["qa_result"] = None
                st.warning(
                    "当前资料中没有找到足够证据。\n\n"
                    "可以换一种问法，或者确认论文里真的写了这个信息。"
                )
            else:
                with st.spinner("AI 正在读你给的原文…"):
                    try:
                        answer = ask_deepseek(api_key, question, hits, compare=compare)
                    except Exception as exc:
                        st.error(f"调用失败：{exc}")
                        return
                st.session_state["qa_result"] = {"answer": answer, "hits": hits}

    result = st.session_state.get("qa_result")
    if result:
        st.markdown("### 回答")
        st.markdown(result["answer"])
        with st.expander("查看 AI 实际读到的原文（建议评委面前点开）"):
            for i, hit in enumerate(result["hits"], start=1):
                st.markdown(f"**资料 {i} · {hit.file} · 第 {hit.page} 页**")
                st.text(hit.text[:1200])


def render_translate(api_key, pages):
    st.subheader("论文翻译")
    st.caption("按页翻译，保留公式、术语和引用编号；适合读英文测试论文。")

    if not pages:
        st.info("当前没有可翻译的页。")
        return

    files = sorted({p.file for p in pages})
    f = st.selectbox("选择论文", files, key="trans_file")
    page_nums = sorted({p.page for p in pages if p.file == f})
    pg = st.selectbox("选择页码", page_nums, key="trans_page")
    target = st.radio("翻译成", ["中文", "English"], horizontal=True, key="trans_target")

    page_text = next(p.text for p in pages if p.file == f and p.page == pg)
    with st.expander("先看原文"):
        st.text(page_text[:3000])

    if st.button("翻译这一页", type="primary", key="trans_btn"):
        if not api_key.strip():
            st.warning("左侧先填 DeepSeek API Key。")
            return
        text = page_text[:5000]
        with st.spinner("正在翻译…"):
            try:
                translated = translate_text(api_key, text, target=target)
            except Exception as exc:
                st.error(f"翻译失败：{exc}")
                return
        st.session_state["trans_result"] = translated

    translated = st.session_state.get("trans_result")
    if translated:
        st.markdown("### 译文")
        st.markdown(translated)
        st.download_button(
            "下载译文（Markdown）",
            translated,
            file_name="translation.md",
            mime="text/markdown",
        )


def render_graph(pages):
    st.subheader("引用图谱（轻量版）")
    st.caption("从论文 References 列表抽取“论文 → 参考文献”的关系；不做跨库引用网络。")

    if st.button("生成引用图谱", type="primary", key="graph_btn"):
        refs = extract_references(pages, max_refs=18)
        st.session_state["graph_refs"] = refs
        if not refs:
            st.warning(
                "没有解析到参考文献。可能这篇 PDF 的 References 是图片/特殊排版；"
                "换一篇文字版论文再试。"
            )

    refs = st.session_state.get("graph_refs", [])
    if refs:
        papers = sorted({p.file for p in pages})
        svg, height = build_citation_svg(papers, refs)
        components.html(svg, height=height, scrolling=True)
        st.caption("图：左侧为上传论文，右侧为它引用的参考文献；线条表示引用关系。")

        rows = [
            {"论文": f, "页码": pg, "参考文献": text[:140]}
            for f, pg, text in refs
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_writing(api_key, chunks):
    st.subheader("写论述")
    st.caption("基于上传论文生成 Related Work / 研究现状草稿；每个事实句都要带页码。")

    topic = st.text_input(
        "论述主题",
        placeholder="例如：LLM 生成单元测试的评估方法与局限",
        key="write_topic",
    )
    style = st.selectbox(
        "写作类型",
        ["Related Work（相关工作）", "研究现状综述", "答辩讲稿提纲"],
        key="write_style",
    )

    if st.button("生成论述", type="primary", key="write_btn"):
        if not topic.strip():
            st.warning("先写一个主题。")
        elif not api_key.strip():
            st.warning("左侧先填 DeepSeek API Key。")
        else:
            hits, top_score = search(topic, chunks, top_k=8)
            if not hits:
                st.warning("没有检索到相关原文，换一个更具体的主题。")
            else:
                with st.spinner("正在组织论述…"):
                    try:
                        text = write_discussion(api_key, topic, hits, style=style)
                    except Exception as exc:
                        st.error(f"生成失败：{exc}")
                        return
                st.session_state["write_result"] = text

    text = st.session_state.get("write_result")
    if text:
        st.markdown("### 论述草稿")
        st.markdown(text)
        st.download_button(
            "下载论述（Markdown）",
            text,
            file_name="related_work.md",
            mime="text/markdown",
        )
        st.caption("AI 生成内容，提交前请逐句核对页码和原文。")


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("设置")
    api_key = st.text_input(
        "DeepSeek API Key",
        value=default_key(),
        type="password",
        help="去 platform.deepseek.com 创建。只用于本次会话，不要发给别人。",
    )
    compare = st.checkbox(
        "多论文对比模式", value=False,
        help="问答时尽量用表格对比；问题里写清楚对比哪几篇。",
    )
    top_k = st.slider("每次检索的原文段数", 3, 10, 6)
    if st.button("清空当前论文", use_container_width=True):
        clear_all()
        st.success("已清空，请重新上传论文。")
    st.divider()
    st.caption("Key 不会写进代码；不要把 Key 提交到 GitHub。")

# ---------------- header ----------------
st.title("查找小助手")
st.caption("面向软件测试论文的证据型阅读助手 · 问答 / 翻译 / 引用图谱 / 写论述")
st.info(
    "上传 1–5 篇**文字版** PDF，先建索引；然后可以用四个功能："
    "**问答（带页码）、翻译、引用图谱、写论述**。扫描件暂不支持 OCR。"
)

files = st.file_uploader(
    "上传 PDF（建议 1–5 篇文字版论文）",
    type=["pdf"],
    accept_multiple_files=True,
    help="扫描件 PDF 暂时读不出文字；先用文字版论文做 Demo。",
)

col_a, col_b = st.columns([1, 2])
with col_a:
    build = st.button("建立索引", type="primary", use_container_width=True)
with col_b:
    st.caption("建议每篇 5–30 页；第一次读 3 篇大约 10–60 秒。")

if build:
    if not files:
        st.warning("先上传至少 1 篇 PDF。")
    else:
        all_pages, empty_pages = [], 0
        progress = st.progress(0.0, text="正在读取 PDF…")
        failed = []
        for i, f in enumerate(files):
            try:
                pages, empty = extract_pages(f.read(), f.name)
                all_pages.extend(pages)
                empty_pages += empty
            except Exception as exc:
                failed.append(f"{f.name}: {exc}")
            progress.progress((i + 1) / len(files), text=f"已读取 {i + 1}/{len(files)} 篇")
        progress.empty()

        if failed:
            st.error("这些文件读取失败：\n\n- " + "\n- ".join(failed))
        if not all_pages:
            st.error("没有读到任何文字。请换文字版 PDF，或确认文件没有加密。")
        else:
            chunks = chunk_pages(all_pages)
            st.session_state["chunks"] = chunks
            st.session_state["pages"] = all_pages
            st.session_state["stats"] = {
                "files": len(files),
                "pages": len(all_pages),
                "chunks": len(chunks),
                "empty": empty_pages,
            }
            for k in ("qa_result", "trans_result", "graph_refs", "write_result"):
                st.session_state.pop(k, None)
            st.success(
                f"索引完成：{len(files)} 篇论文，{len(all_pages)} 页，{len(chunks)} 个原文段。"
            )

if "stats" in st.session_state:
    stats = st.session_state["stats"]
    c1, c2, c3 = st.columns(3)
    c1.metric("论文", stats["files"])
    c2.metric("有文字的页", stats["pages"])
    c3.metric("原文段", stats["chunks"])
    if stats["empty"]:
        st.warning(
            f"有 {stats['empty']} 页没有读到文字，可能是扫描件。"
            "当前版本不支持 OCR，演示时请换文字版 PDF。"
        )

if "chunks" in st.session_state:
    chunks = st.session_state["chunks"]
    pages = st.session_state.get("pages", [])
    tabs = st.tabs(["问答", "翻译", "引用图谱", "写论述"])
    with tabs[0]:
        render_qa(api_key, chunks, top_k, compare)
    with tabs[1]:
        render_translate(api_key, pages)
    with tabs[2]:
        render_graph(pages)
    with tabs[3]:
        render_writing(api_key, chunks)
else:
    st.info("先上传论文并点“建立索引”，四个功能会解锁。")

st.divider()
st.caption(
    "测试用 Demo，不保证答案 100% 正确；所有结论请点开原文核对。"
    "不要上传未公开论文、隐私或涉密材料。"
)
