# -*- coding: utf-8 -*-
"""TestPaper Copilot 保命版：一个能演示的论文问答网页。"""

import os
import streamlit as st

from core import extract_pages, chunk_pages, search, ask_deepseek


st.set_page_config(
    page_title="TestPaper Copilot · 保命版",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1100px;}
    h1 {color: #0B3D33;}
    .small-note {color: #6F7C76; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# 云端部署时把 Key 放在 Streamlit Secrets；本地可以直接在左侧输入。
try:
    secret_key = st.secrets.get("DEEPSEEK_API_KEY", "")
except Exception:
    secret_key = ""
env_key = os.getenv("DEEPSEEK_API_KEY", "")
default_key = str(secret_key or env_key)

with st.sidebar:
    st.header("设置")
    api_key = st.text_input(
        "DeepSeek API Key",
        value=default_key,
        type="password",
        help="去 platform.deepseek.com 创建。只用于本次会话，不要发给别人。",
    )
    compare = st.checkbox("多论文对比模式", value=False,
                          help="开启后回答会尽量用表格对比；问题里写清楚对比哪几篇。")
    top_k = st.slider("每次检索的原文段数", 3, 10, 6)
    if st.button("清空当前论文", use_container_width=True):
        for k in ("chunks", "stats"):
            st.session_state.pop(k, None)
        st.success("已清空，请重新上传论文。")
    st.divider()
    st.caption("Key 不会写进代码；不要把 Key 提交到 GitHub。")

st.title("TestPaper Copilot · 保命版")
st.caption("上传论文 → 提一个问题 → 回答必须标出 [文件名 第X页]")
st.info(
    "这个版本只做三件事：**读 PDF、回答带页码、找不到就说没有证据**。"
    "翻译、引用图谱、写综述先不做。先把这一条线跑通。"
)

files = st.file_uploader(
    "上传 PDF（建议 2–5 篇文字版论文）",
    type=["pdf"],
    accept_multiple_files=True,
    help="扫描件 PDF 暂时读不出文字；先用文字版论文做 Demo。",
)

col_a, col_b = st.columns([1, 2])
with col_a:
    build = st.button("建立索引", type="primary", use_container_width=True)
with col_b:
    st.caption("建议每篇 5–30 页；第一次读 5 篇大约 10–60 秒。")

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
            st.session_state["stats"] = {
                "files": len(files),
                "pages": len(all_pages),
                "chunks": len(chunks),
                "empty": empty_pages,
            }
            st.success(f"索引完成：{len(files)} 篇论文，{len(all_pages)} 页，{len(chunks)} 个原文段。")

if "stats" in st.session_state:
    stats = st.session_state["stats"]
    c1, c2, c3 = st.columns(3)
    c1.metric("论文", stats["files"])
    c2.metric("有文字的页", stats["pages"])
    c3.metric("原文段", stats["chunks"])
    if stats["empty"]:
        st.warning(
            f"有 {stats['empty']} 页没有读到文字，可能是扫描件。"
            "这个保命版不支持 OCR，演示时请换文字版 PDF。"
        )

with st.expander("可以问什么？点开看示例"):
    st.markdown(
        """
- 哪篇论文在 Defects4J 上报告了 mutation score？给出页码。
- 这几篇论文分别用了什么数据集和指标？做一个对比表。
- 这篇论文的威胁效度（threats to validity）是什么？
- 这篇论文有没有公开代码和数据集？链接在哪？
- 这篇论文的工业落地客户是谁？（预期：拒答，论文里没写）
        """
    )

st.divider()
question = st.text_input(
    "问论文里的问题",
    placeholder="例如：哪篇论文在 Defects4J 上报告了 mutation score？",
)
ask = st.button("提问", type="primary")

if ask:
    if "chunks" not in st.session_state:
        st.warning("先点上面的“建立索引”。")
    elif not question.strip():
        st.warning("先写一个问题。")
    elif not api_key.strip():
        st.warning("左侧先填 DeepSeek API Key。没有 Key 就无法回答。")
    else:
        chunks = st.session_state["chunks"]
        k = max(top_k, 10) if compare else top_k
        hits, top_score = search(question, chunks, top_k=k)

        if not hits or top_score < 0.08:
            st.warning(
                "当前资料中没有找到足够证据。\n\n"
                "可以换一种问法，或者确认论文里真的写了这个信息。"
            )
        else:
            with st.spinner("AI 正在读你给的原文…"):
                try:
                    answer = ask_deepseek(
                        api_key, question, hits, compare=compare
                    )
                except Exception as exc:
                    st.error(f"调用失败：{exc}")
                    st.stop()

            st.markdown("### 回答")
            st.markdown(answer)

            with st.expander("查看 AI 实际读到的原文（建议评委面前点开）"):
                for i, hit in enumerate(hits, start=1):
                    st.markdown(f"**资料 {i} · {hit.file} · 第 {hit.page} 页**")
                    st.text(hit.text[:1200])

st.divider()
st.caption(
    "测试用 Demo，不保证答案 100% 正确；所有结论请点开原文核对。"
    "不要上传未公开论文、隐私或涉密材料。"
)
