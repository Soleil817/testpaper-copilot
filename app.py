# -*- coding: utf-8 -*-
"""查找小助手：面向软件测试论文的证据型阅读助手。"""

import os
import json
import streamlit as st
import streamlit.components.v1 as components

from core import (
    extract_pages, chunk_pages, search, ask_deepseek, ask_deepseek_fix,
    ask_deepseek_stream, translate_text, write_discussion, extract_references,
    verify_answer, find_duplicate_groups, diff_versions, format_diff, expand_hits,
    arxiv_search, download_arxiv_pdf, render_page_png, search_multi,
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
              "graph_refs", "write_result", "qa_log", "archive_pages",
              "dup_groups", "diff_report", "pdf_bytes", "trans_cache",
              "reader_file", "reader_page"):
        st.session_state.pop(k, None)


STATUS = {"ok": "通过", "weak": "待核对", "missing": "页码未命中"}


def _log_qa(question, answer, hits, report=None, refused=False, top_score=0.0):
    """把每次问答写进会话日志；导出 JSONL 后可直接跑 eval/score.py。"""
    if "qa_log" not in st.session_state:
        st.session_state["qa_log"] = []
    log = st.session_state["qa_log"]
    report = report or {}
    log.append({
        "id": len(log) + 1,
        "question": question,
        "answer": answer,
        "top_score": round(float(top_score or 0.0), 3),
        "hits": [[h.file, h.page] for h in hits],
        "refused": refused or report.get("refused", False),
        "pass": report.get("pass", False),
        "ok": report.get("ok", 0),
        "weak": report.get("weak", 0),
        "missing": report.get("missing", 0),
        "citations": report.get("citations", []),
    })


def add_to_library(new_pages, file_name=None, pdf_bytes=None):
    """把新论文并进当前文献库，重建索引与去重检查。"""
    pages = list(st.session_state.get("pages") or []) + list(new_pages)
    st.session_state["pages"] = pages
    st.session_state["archive_pages"] = pages
    st.session_state["chunks"] = chunk_pages(pages)
    if file_name and pdf_bytes:
        store = dict(st.session_state.get("pdf_bytes") or {})
        store[file_name] = pdf_bytes
        st.session_state["pdf_bytes"] = store
    old = st.session_state.get("stats") or {}
    st.session_state["stats"] = {
        "files": len({p.file for p in pages}),
        "pages": len(pages),
        "chunks": len(st.session_state["chunks"]),
        "empty": old.get("empty", 0),
    }
    by_file = {}
    for p in pages:
        by_file.setdefault(p.file, []).append(p)
    st.session_state["dup_groups"] = find_duplicate_groups(by_file)


def render_arxiv():
    """模块四：调用 arXiv 公开 API 检索并入库（工具调用入口）。"""
    st.caption(
        "调用 arXiv 公开 API 检索论文，直接下载 PDF 入库；摘要/备注里出现 GitHub 链接会标出来"
        "（复现性提示）。这一块对应赛题模块四，也是“Agent 调工具”的最小演示。"
    )
    c1, c2 = st.columns([3, 1])
    query = c1.text_input(
        "关键词 / arXiv ID", key="arxiv_q",
        placeholder="例如 2305.04764，或 LLM test generation",
    )
    num = c2.number_input("条数", 1, 10, 5, key="arxiv_n")

    if st.button("搜索 arXiv", key="arxiv_search_btn"):
        try:
            with st.spinner("正在调用 arXiv API…"):
                results = arxiv_search(query, int(num))
        except Exception as exc:
            st.error(f"检索失败：{exc}")
        else:
            st.session_state["arxiv_results"] = results
            if not results:
                st.info("没有结果，换个关键词，或直接贴 arXiv ID。")

    results = st.session_state.get("arxiv_results") or []
    for r in results:
        with st.expander(f"{r['id']} · {r['title'][:70]}"):
            st.markdown(f"**作者**：{r['authors'] or '—'}")
            st.caption(f"发布：{r['published']}　PDF：{r['pdf_url']}")
            st.write(r["summary"][:700])
            if r["github"]:
                st.success(f"检测到开源代码（复现性较好）：{r['github']}")
            else:
                st.caption("摘要里没有 GitHub 链接，复现性需要人工确认。")

            if st.button("下载并加入文献库", key=f"arxiv_add_{r['id']}"):
                fname = f"{r['id']}.pdf"
                try:
                    with st.spinner(f"正在下载 {fname} …"):
                        pdf = download_arxiv_pdf(r["pdf_url"])
                        new_pages, _ = extract_pages(pdf, fname)
                except Exception as exc:
                    st.error(f"下载失败：{exc}")
                else:
                    if not new_pages:
                        st.warning("这篇没读到文字，可能是扫描版；当前版本不解析扫描件。")
                    elif any(p.file == fname for p in st.session_state.get("pages") or []):
                        st.info(f"{fname} 已经在文献库里了。")
                    else:
                        add_to_library(new_pages, fname, pdf)
                        st.success(f"已加入 {fname}（{len(new_pages)} 页），索引已重建。")
                        st.rerun()


def render_check(report):
    """把“标了页码”升级成“页码经程序校验”。评委最吃这一口。"""
    if report.get("refused"):
        st.info("系统判断：资料中没有足够证据，已明确拒答（不编造）。")
        return
    if not report.get("total"):
        st.warning("回答里没有检测到 [文件名 第X页] 形式的引用，无法校验。")
        return

    if report["pass"]:
        st.success(
            f"引用校验通过：{report['ok']}/{report['total']} 条引用都能在本次检索到的原文中定位，"
            "且词面有支撑。"
        )
    elif report["missing"]:
        st.error(
            f"{report['missing']} 处引用的页码**不在本次检索到的原文里**（疑似编造），"
            f"{report['weak']} 处支撑较弱。建议点下方“修复未通过的引用”。"
        )
    else:
        st.warning(f"页码都能对上，但有 {report['weak']} 条引用词面支撑较弱，建议人工核对。")

    rows = [
        {
            "结论": (i["claim"] or "（空）")[:90],
            "引用": i["raw"],
            "状态": STATUS.get(i["status"], i["status"]),
            "支撑度": "-" if i["support"] is None else f"{i['support']:.2f}",
            "原文证据句": (i["evidence"] or "—")[:120],
        }
        for i in report["citations"]
    ]
    with st.expander("逐条校验明细（程序化核对，不是模型自说自话）"):
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "规则：先查页码是否真的在这次检索到的原文里；再比这句话与原文的实词重合度。"
            "中文结论 + 英文论文时只比英文术语，避免误判。"
        )
        quotes = [i for i in report["citations"] if i.get("evidence")]
        for i in quotes[:5]:
            st.markdown(f"> {i['evidence']}")
            st.caption(f"— {i['file']} 第 {i['page']} 页，用于支撑：{(i['claim'] or '')[:60]}")


def render_qa(api_key, chunks, top_k, compare, retrieval):
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
            hits_raw, top_score = search_multi(question, chunks, top_k=k, mode=retrieval,
                                         per_file=2 if compare else 0)
            hits = expand_hits(hits_raw, chunks) if hits_raw else []
            if not hits or top_score < 0.05:
                st.session_state["qa_result"] = None
                _log_qa(question, "", [], refused=True, top_score=top_score)
                st.warning(
                    "当前资料中没有找到足够证据。\n\n"
                    "可以换一种问法，或者确认论文里真的写了这个信息。"
                )
            else:
                st.markdown("### 回答")
                st.caption(f"检索到 {len(hits_raw)} 段原文（已扩展为整页 {len(hits)} 段），"
                           "正在流式生成并逐条校验…")
                try:
                    answer = st.write_stream(
                        ask_deepseek_stream(api_key, question, hits, compare=compare)
                    )
                except Exception as exc:
                    st.error(f"调用失败：{exc}")
                    return
                if not (answer or "").strip():
                    st.error("模型没有返回内容，请重试。")
                    return
                report = verify_answer(answer, hits)
                st.session_state["qa_result"] = {
                    "answer": answer, "hits": hits, "report": report, "question": question,
                }
                st.session_state["qa_streamed"] = True
                _log_qa(question, answer, hits_raw, report=report, top_score=top_score)

    result = st.session_state.get("qa_result")
    if result:
        report = result.get("report") or verify_answer(result["answer"], result["hits"])
        if not st.session_state.pop("qa_streamed", False):
            st.markdown("### 回答")
            st.markdown(result["answer"])
        render_check(report)

        cites = report.get("citations") or []
        if cites:
            options = {f"{c['file']} · 第{c['page']}页": (c["file"], c["page"]) for c in cites}
            choice = st.selectbox("定位到引用原文", list(options), key="jump_sel")
            if st.button("在「阅读器」打开", key="jump_btn"):
                target_file, target_page = options[choice]
                st.session_state["reader_file"] = target_file
                st.session_state["reader_page"] = target_page
                st.session_state["reader_hint"] = f"已从问答定位：{target_file} 第 {target_page} 页"
                st.rerun()

        if not report["pass"] and not report["refused"]:
            if st.button("修复未通过的引用（只重写这几句）", key="qa_fix"):
                with st.spinner("正在按原文修订未通过的句子…"):
                    try:
                        fixed = ask_deepseek_fix(
                            api_key, result["question"], result["hits"],
                            result["answer"], report,
                        )
                        new_report = verify_answer(fixed, result["hits"])
                    except Exception as exc:
                        st.error(f"修复失败：{exc}")
                    else:
                        st.session_state["qa_result"] = {
                            "answer": fixed, "hits": result["hits"],
                            "report": new_report, "question": result["question"],
                        }
                        st.rerun()

        with st.expander("查看 AI 实际读到的原文（建议评委面前点开）", expanded=not report["pass"]):
            for i, hit in enumerate(result["hits"], start=1):
                st.markdown(f"**资料 {i} · {hit.file} · 第 {hit.page} 页**")
                st.text(hit.text[:1200])

        if st.session_state.get("qa_log"):
            st.download_button(
                "导出本次问答记录（JSONL，评测用）",
                json.dumps(st.session_state["qa_log"], ensure_ascii=False, indent=1),
                file_name="qa_log.jsonl",
                mime="application/json",
                help="拿去填 eval/ 里的评测表，算引用通过率、拒答准确率。",
            )


def render_reader(api_key, pages):
    """模块三：左原文版式、右译文的按页联动阅读。"""
    st.subheader("阅读器：左侧原版式，右侧译文（按页联动）")
    st.caption(
        "左栏是 PDF 原页渲染，排版/公式/表格保持原样；右栏按页给译文。翻页两边同步，"
        "从问答的引用也能直接定位过来。"
    )

    if not pages:
        st.info("先上传论文并建立索引。")
        return

    hint = st.session_state.pop("reader_hint", "")
    if hint:
        st.success(hint)

    files = sorted({p.file for p in pages})
    if st.session_state.get("reader_file") not in files:
        st.session_state["reader_file"] = files[0]
    file = st.selectbox("论文", files, key="reader_file")

    page_nums = sorted({p.page for p in pages if p.file == file})
    if st.session_state.get("reader_page") not in page_nums:
        st.session_state["reader_page"] = page_nums[0]
    page = st.session_state["reader_page"]
    idx = page_nums.index(page)

    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
    if c1.button("← 上一页", key="reader_prev", disabled=idx == 0):
        st.session_state["reader_page"] = page_nums[idx - 1]
        st.rerun()
    if c2.button("下一页 →", key="reader_next", disabled=idx == len(page_nums) - 1):
        st.session_state["reader_page"] = page_nums[idx + 1]
        st.rerun()
    c3.caption(f"第 {page} / {len(page_nums)} 页")
    mode = c4.radio("右侧显示", ["译文", "原文", "对照"], horizontal=True, key="reader_mode")

    left, right = st.columns([1, 1])
    with left:
        png = render_page_png((st.session_state.get("pdf_bytes") or {}).get(file), page)
        if png:
            st.image(png, caption=f"{file} · 第 {page} 页（原版式）", use_container_width=True)
        else:
            st.info("这一页没有原始 PDF 图像。重新建立索引后会保存原始文件；"
                    "新上传或从 arXiv 入库的论文都有图。")

    with right:
        page_text = next((p.text for p in pages if p.file == file and p.page == page), "")
        if mode in ("原文", "对照"):
            st.markdown("**原文**")
            st.text(page_text[:3000])
        if mode in ("译文", "对照"):
            st.markdown("**译文**")
            if "trans_cache" not in st.session_state:
                st.session_state["trans_cache"] = {}
            cache = st.session_state["trans_cache"]
            key = (file, page)
            if key in cache:
                st.markdown(cache[key])
            elif st.button("翻译本页（会缓存）", key="reader_trans"):
                if not api_key.strip():
                    st.warning("左侧先填 DeepSeek API Key。")
                else:
                    with st.spinner("翻译中…"):
                        try:
                            cache[key] = translate_text(api_key, page_text[:5000])
                        except Exception as exc:
                            st.error(f"翻译失败：{exc}")
                    st.rerun()
            else:
                st.caption("点“翻译本页”生成；同一页只翻一次，翻回来直接看缓存。")


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
    retrieval = st.selectbox(
        "检索方式",
        ["hybrid", "bm25", "coverage"],
        format_func=lambda m: {"hybrid": "混合（BM25 + 词面覆盖）",
                               "bm25": "BM25（稀有术语优先）",
                               "coverage": "词面覆盖（旧版）"}[m],
        help="换一种排序，用同一批问题对比命中率——这就是文档里的“方案对比”实验。",
    )
    if st.button("清空当前论文", use_container_width=True):
        clear_all()
        st.success("已清空，请重新上传论文。")
    st.divider()
    st.caption("Key 不会写进代码；不要把 Key 提交到 GitHub。")
    st.caption("build 2026-11-12 · multi-query")

# ---------------- header ----------------
st.title("查找小助手")
st.caption("面向软件测试论文的证据型阅读助手 · 问答 / 引用校验 / 双栏阅读 / 引用图谱 / 写论述")
st.info(
    "上传 1–5 篇**文字版** PDF，先建索引；然后可以用：**问答（带页码）、引用校验、"
    "双栏阅读（左原文右译文）、引用图谱、写论述**；首页还能从 arXiv 直接检索入库。"
    "扫描件暂不支持 OCR。"
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
        pdf_map = {}
        progress = st.progress(0.0, text="正在读取 PDF…")
        failed = []
        for i, f in enumerate(files):
            try:
                raw = f.read()
                pages, empty = extract_pages(raw, f.name)
                all_pages.extend(pages)
                empty_pages += empty
                pdf_map[f.name] = raw
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
            st.session_state["archive_pages"] = all_pages
            st.session_state["pdf_bytes"] = pdf_map
            st.session_state["stats"] = {
                "files": len(files),
                "pages": len(all_pages),
                "chunks": len(chunks),
                "empty": empty_pages,
            }
            by_file = {}
            for p in all_pages:
                by_file.setdefault(p.file, []).append(p)
            st.session_state["dup_groups"] = find_duplicate_groups(by_file)
            st.session_state.pop("diff_report", None)
            for k in ("qa_result", "trans_result", "graph_refs", "write_result"):
                st.session_state.pop(k, None)
            st.success(
                f"索引完成：{len(files)} 篇论文，{len(all_pages)} 页，{len(chunks)} 个原文段。"
            )

with st.expander("arXiv 检索入库（工具调用）", expanded=False):
    render_arxiv()

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

    pages_now = st.session_state.get("pages") or []
    by_name = {}
    for p in pages_now:
        d = by_name.setdefault(p.file, {"文件": p.file, "页数": 0, "原文段": 0})
        d["页数"] += 1
    chunk_counts = {}
    for c in st.session_state.get("chunks") or []:
        chunk_counts[c.file] = chunk_counts.get(c.file, 0) + 1
    for name, d in by_name.items():
        d["原文段"] = chunk_counts.get(name, 0)
    with st.expander(f"已入库论文（{len(by_name)} 篇）", expanded=False):
        st.dataframe(list(by_name.values()), use_container_width=True, hide_index=True)


def render_dedupe():
    """官方测试用例 3.1：按内容识别同一论文的不同版本，并总结差异。"""
    groups = st.session_state.get("dup_groups") or []
    if not groups:
        return
    st.subheader("文献库检查：语义去重")
    for g in groups:
        names = "、".join(g["files"])
        st.warning(
            f"检测到内容高度重合的论文（相似度 {g['similarity']:.2f}）：{names}\n\n"
            f"建议保留 **{g['keep']}**"
            + (f"（arXiv:{g['arxiv_id']}）" if g["arxiv_id"] else "")
            + "，其余版本可选保留或排除。"
        )

    archive = st.session_state.get("archive_pages") or []
    names = sorted({p.file for p in archive})
    with st.expander("对比两个版本的差异（官方测试用例 3.1）"):
        c1, c2 = st.columns(2)
        old_name = c1.selectbox("旧版本", names, index=0, key="diff_old")
        new_name = c2.selectbox("新版本", names, index=len(names) - 1, key="diff_new")
        if st.button("生成版本差异摘要", key="diff_btn"):
            if old_name == new_name:
                st.warning("选了两个相同的文件。")
            else:
                old_pages = [p for p in archive if p.file == old_name]
                new_pages = [p for p in archive if p.file == new_name]
                st.session_state["diff_report"] = format_diff(
                    diff_versions(old_pages, new_pages), old_name, new_name
                )
        report = st.session_state.get("diff_report")
        if report:
            st.markdown(report)
            st.caption("按句子粒度对比，页码是各版本自己的页码；只列前几处，完整数据可用脚本导出。")

    drop = sorted({f for g in groups for f in g["files"] if f != g["keep"]})
    if drop and st.button(f"只保留较新版本并重建索引（排除 {len(drop)} 篇）", key="dedupe_keep"):
        stats = dict(st.session_state["stats"])
        kept = [p for p in st.session_state["pages"] if p.file not in drop]
        st.session_state["pages"] = kept
        st.session_state["chunks"] = chunk_pages(kept)
        stats.update({"files": stats["files"] - len(drop), "pages": len(kept),
                      "chunks": len(st.session_state["chunks"])})
        st.session_state["stats"] = stats
        st.session_state["dup_groups"] = []
        st.session_state.pop("diff_report", None)
        st.rerun()


if "stats" in st.session_state:
    render_dedupe()

if "chunks" in st.session_state:
    chunks = st.session_state["chunks"]
    pages = st.session_state.get("pages", [])
    tabs = st.tabs(["问答", "阅读器", "引用图谱", "写论述"])
    with tabs[0]:
        render_qa(api_key, chunks, top_k, compare, retrieval)
    with tabs[1]:
        render_reader(api_key, pages)
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
