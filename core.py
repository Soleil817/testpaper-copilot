# -*- coding: utf-8 -*-
"""TestPaper Copilot 保命版：PDF 读取、切块、检索、带页码回答。"""

from dataclasses import dataclass
import json
import os
import re
import time


@dataclass
class Page:
    file: str
    page: int
    text: str


@dataclass
class Chunk:
    file: str
    page: int
    text: str


# 中文按“两个字”切，英文按单词切；够这个 Demo 用了。
CJK = re.compile(r"[\u4e00-\u9fff]")
WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-.]{1,}")
STOP = set(
    "的 了 是 在 和 与 及 对 中 有 为 上 也 这 那 我 你 他 她 它 们 "
    "吗 呢 啊 什么 怎么 哪些 哪篇 哪个 多少 请 帮 一下".split()
)


def clean_text(text: str) -> str:
    text = (text or "").replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_bytes: bytes, filename: str):
    """返回 (有文字的页, 没有文字的页数)。扫描件会落到“没有文字”。"""
    import fitz  # PyMuPDF，放在函数里，方便没装依赖时给出友好提示

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages, empty = [], 0
    for i, page in enumerate(doc, start=1):
        text = clean_text(page.get_text("text"))
        if len(text) < 20:
            empty += 1
            continue
        pages.append(Page(file=filename, page=i, text=text))
    doc.close()
    return pages, empty


def chunk_pages(pages, size: int = 900, overlap: int = 150):
    """按页切块，保留文件名和页码，后面才能给出处。"""
    chunks = []
    for p in pages:
        text = p.text
        if len(text) <= size:
            chunks.append(Chunk(p.file, p.page, text))
            continue
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(Chunk(p.file, p.page, text[start:end]))
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks


def token_counts(text: str):
    """tokenize 的计数版：BM25 需要词频，不能只用 set。"""
    text = (text or "").lower()
    counts = {}

    for w in WORD.findall(text):
        if len(w) >= 2 and w not in STOP:
            counts[w] = counts.get(w, 0) + 1
            # 论文里的 d_model 常被抽成 dmodel；两种写法都当作同一个词
            if "_" in w:
                flat = w.replace("_", "")
                counts[flat] = counts.get(flat, 0) + 1

    chars = [c for c in text if CJK.match(c)]
    for i in range(len(chars) - 1):
        bigram = chars[i] + chars[i + 1]
        if bigram not in STOP:
            counts[bigram] = counts.get(bigram, 0) + 1

    return counts


# 中文提问、英文原文时，光靠中文二字组永远匹配不上。
# 这里做一层最小的“查询扩展”：命中中文术语就在检索词里补上英文写法。
CN2EN = {
    "编码器": "encoder", "解码器": "decoder", "注意力": "attention",
    "多头": "multi-head", "头数": "heads", "几层": "layers", "层数": "layers",
    "隐藏": "hidden", "前馈": "feed-forward", "位置编码": "positional encoding",
    "嵌入": "embedding", "超参数": "hyperparameter", "参数": "parameters",
    "训练": "training", "数据集": "dataset", "指标": "metric evaluation coverage accuracy", "基线": "baseline comparison",
    "消融": "ablation", "实验": "experiment", "评估": "evaluation evaluate",
    "缺陷": "defects", "变异": "mutation", "测试用例": "test cases",
    "生成": "generation", "模型": "model", "局限": "limitations",
    "威胁效度": "threats to validity", "未来工作": "future work",
    "贡献": "contribution", "方法": "method approach", "架构": "architecture",
    "复杂度": "complexity", "学习率": "learning rate", "显存": "memory",
    "推理": "inference", "开源": "open source", "复现": "reproduce",
    "参考文献": "references", "表格": "table", "图": "figure",
    "梯度下降": "gradient descent", "反向传播": "backpropagation",
    "过拟合": "overfitting", "正则化": "regularization", "卷积": "convolution",
    "池化": "pooling", "激活": "activation", "优化器": "optimizer",
    "损失": "loss", "泛化": "generalization", "残差": "residual",
    "归一化": "normalization", "动量": "momentum", "批归一化": "batch normalization",
    "问题": "problem issue challenge address", "挑战": "challenge", "结论": "conclusion results finding", "任务": "task",
    "目标": "goal aim objective", "结果": "results findings", "效果": "performance", "质量": "quality",
    "覆盖率": "coverage", "准确率": "accuracy", "效率": "efficiency",
    "工具": "tool framework", "代码": "code", "评价": "evaluation evaluate", "贡献": "contribution",
}


def expand_query(question: str) -> str:
    """把中文术语翻成英文检索词；只影响召回，不影响回答语言。"""
    extra = [en for cn, en in CN2EN.items() if cn in (question or "")]
    return " ".join(dict.fromkeys(extra))


def tokenize(text: str):
    return set(token_counts(text))


def score_chunk(q_tokens, question: str, chunk: Chunk) -> float:
    """词面覆盖率（0–1），保留原语义：解释性强、可当拒答阈值。"""
    if not q_tokens:
        return 0.0
    doc_tokens = tokenize(chunk.file + " " + chunk.text)
    overlap = sum(1 for t in q_tokens if t in doc_tokens)
    coverage = overlap / max(1, len(q_tokens))

    # 英文术语（如 Defects4J / mutation score）额外加一点分
    bonus = 0.0
    for w in WORD.findall(question.lower()):
        if len(w) >= 3 and w in chunk.text.lower():
            bonus += 0.05
    return coverage + min(bonus, 0.2)


def _bm25_scores(q_tokens, chunks, k1: float = 1.5, b: float = 0.75):
    """纯 Python 的 Okapi BM25：不引入新依赖，部署不会因为装包失败。

    关键词法把“论文/实验”这类到处都有的词和“Defects4J”一样算分；
    BM25 用 IDF 压掉高频词、按文档长度归一，稀有术语更容易冒出来。
    """
    import math
    from collections import Counter

    docs = [token_counts(c.file + " " + c.text) for c in chunks]
    n = len(docs)
    if not n:
        return []

    df = Counter()
    for d in docs:
        df.update(d.keys())
    avgdl = sum(sum(d.values()) for d in docs) / n or 1.0
    q = list(q_tokens)

    out = []
    for d in docs:
        dl = sum(d.values()) or 1
        score = 0.0
        for t in q:
            tf = d.get(t, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
        out.append(score)
    return out


GENERIC_MARKERS = ("核心问题", "解决什么问题", "研究的是什么", "研究什么", "主要结论",
                   "贡献", "motivation", "研究任务")


def ensure_all_files(hits, chunks):
    """对比模式：没命中的论文补第 1 页，保证每篇都有代表。

    否则检索分 0 的论文会被整篇丢掉，模型只能对比其中一两篇。
    """
    have = {h.file for h in hits}
    extra = []
    for c in chunks:
        if c.file not in have and c.page == 1:
            extra.append(c)
            have.add(c.file)
    return list(hits) + extra


def with_first_page(question: str, chunks, hits):
    """问题类提问强制带上第 1 页（摘要/引言），否则摘要页常被实现页挤掉。"""
    if not any(m in (question or "") for m in GENERIC_MARKERS):
        return hits
    firsts = [c for c in chunks if c.page == 1]
    if not firsts:
        return hits
    seen = {(c.file, c.page, c.text[:40]) for c in firsts}
    return firsts + [h for h in hits if (h.file, h.page, h.text[:40]) not in seen]


def expand_compare(hits, chunks, max_files: int = 4, max_chunks: int = 16):
    """对比模式专用：每篇论文只展开它最相关的那一页，保证每篇都进上下文。"""
    files = []
    for h in hits:
        if h.file not in files:
            files.append(h.file)
    out = []
    for f in files[:max_files]:
        page = next(h.page for h in hits if h.file == f)
        for c in chunks:
            if c.file == f and c.page == page:
                out.append(c)
                if len(out) >= max_chunks:
                    return out
    return out


def expand_hits(hits, chunks, max_pages: int = 3, max_chunks: int = 12):
    """把命中的原文段扩成整页：同一页的其他段也带上。

    为什么需要：按页切块后，一句话可能被切到另一块里。
    只召回含关键词的那一块，模型就会说“资料未报告”——其实同一页写了。
    """
    pages = []
    for h in hits:
        key = (h.file, h.page)
        if key not in pages:
            pages.append(key)
        if len(pages) >= max_pages:
            break

    out = []
    for key in pages:
        for c in chunks:
            if (c.file, c.page) != key:
                continue
            out.append(c)
            if len(out) >= max_chunks:
                return out
    return out


def _clean_terms(text: str, limit: int = 200) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}", text or "")
    return " ".join(dict.fromkeys(w.lower() for w in words))[:limit]


def rewrite_query(api_key: str, question: str, model: str = "deepseek-chat") -> str:
    """让模型把中文问题改写成英文检索词——解决“数据集/指标”对不上英文原文。

    只影响检索，不影响回答语言；失败时返回空串，调用方退回原问题。
    """
    if not api_key:
        return ""
    system = (
        "你是学术检索助手。把用户的问题改写成 3-6 个英文检索词，"
        "覆盖同义术语、领域惯用词与常见缩写；只输出检索词，用空格分隔，"
        "不要编号、不要解释、不要句子。"
    )
    try:
        raw = _chat(api_key, system, question, model=model, temperature=0, max_tokens=80)
    except Exception:
        return ""
    return _clean_terms(raw)


SPLIT = re.compile(r"以及|[和与及、；;]")


def split_queries(question: str, min_len: int = 4):
    """把“A 和 B 分别是什么”拆成两个子问题。

    一问两件事时，整句算分会偏向其中一件，另一件的原文页召回不到；
    拆开各查一次再合并，是长文档里最划算的修法。
    """
    parts = [p.strip(" ，。,.") for p in SPLIT.split(question or "")]
    return [p for p in parts if len(p) >= min_len]


def search_multi(question: str, chunks, top_k: int = 6, mode: str = "hybrid",
                 per_file: int = 0):
    """多子问题检索：拆句、分别检索、按页去重后合并。"""
    parts = split_queries(question)
    if len(parts) <= 1:
        return search(question, chunks, top_k=top_k, mode=mode, per_file=per_file)

    per = max(2, top_k // len(parts))
    lists, top_score = [], 0.0
    for part in parts:
        hits, score = search(part, chunks, top_k=per, mode=mode, per_file=per_file)
        top_score = max(top_score, score)
        lists.append(hits)

    # 交替合并：A 的第一页、B 的第一页、A 的第二页……
    # 否则扩成整页时前几页全被第一个主题占满，第二个主题会被截掉。
    seen, out = set(), []
    depth = max((len(h) for h in lists), default=0)
    for i in range(depth):
        for hits in lists:
            if i >= len(hits):
                continue
            c = hits[i]
            key = (c.file, c.page, c.text[:40])
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
    return out[: max(top_k, len(parts) * 2)], top_score


def search(question: str, chunks, top_k: int = 6, mode: str = "hybrid",
           per_file: int = 0):
    """返回 (命中的原文段, 最强的词面覆盖率)。

    第二个返回值仍然是覆盖率（0–1），所以 app 里的拒答阈值 0.08 语义不变；
    变的只是排序：mode="coverage" 是旧行为，"bm25" 只看 BM25，
    "hybrid" 两者加权，既能压住高频词，又不会把长页排到前面。

    per_file>0 时按文件轮流取样，保证“多论文对比”每篇都有原文进来。
    """
    q_tokens = tokenize(question) | tokenize(expand_query(question))
    if not q_tokens or not chunks:
        return [], 0.0

    coverage = [score_chunk(q_tokens, question, c) for c in chunks]
    top_score = max(coverage)

    if mode == "coverage":
        ranked = list(range(len(chunks)))
        ranked.sort(key=lambda i: coverage[i], reverse=True)
    else:
        bm25 = _bm25_scores(q_tokens, chunks)
        top_bm = max(bm25) or 1.0
        if mode == "bm25":
            ranked = sorted(range(len(chunks)), key=lambda i: bm25[i], reverse=True)
        else:
            # 归一化后加权：BM25 管“罕见术语”，覆盖率管“问了几个词”
            mixed = [0.65 * (bm25[i] / top_bm) + 0.35 * min(coverage[i], 1.0)
                     for i in range(len(chunks))]
            ranked = sorted(range(len(chunks)), key=lambda i: mixed[i], reverse=True)

    hits = []
    if per_file and per_file > 0:
        # 按文件轮流取：第 1 篇 top1、第 2 篇 top1……再取第 2 轮。
        # 否则排序靠前那篇会把位置占满，后面的论文根本进不了上下文。
        taken, order, by_file = {}, [], {}
        for i in ranked:
            if coverage[i] <= 0:
                continue
            by_file.setdefault(chunks[i].file, []).append(i)
        depth = max((len(v) for v in by_file.values()), default=0)
        for round_idx in range(depth):
            for f, idxs in by_file.items():
                if round_idx >= len(idxs) or taken.get(f, 0) >= per_file:
                    continue
                order.append(idxs[round_idx])
                taken[f] = taken.get(f, 0) + 1
                if len(order) >= top_k:
                    break
            if len(order) >= top_k:
                break
        hits = [chunks[i] for i in order]
    else:
        hits = [chunks[i] for i in ranked[:top_k] if coverage[i] > 0]

    return hits, top_score


def build_prompt(question: str, chunks, compare: bool = False):
    context = "\n\n".join(
        f"【资料{i}】文件：{c.file}  第{c.page}页\n{c.text[:1400]}"
        for i, c in enumerate(chunks, start=1)
    )

    system = (
        "你是软件测试方向的论文阅读助手。"
        "只能使用我给你的资料回答，绝对不要使用外部知识，也不要猜。"
        "每一个事实句的末尾都必须标注 [文件名 第X页]。"
        "如果资料不足以回答，直接回答：当前资料中没有足够证据，并说明缺少什么。"
        "回答用中文，最多 5 条，简洁，不要复述题目。"
    )
    if compare:
        system += (
            " 如果用户要求对比，请输出 Markdown 表格，列为：论文、数据集、指标、"
            "主要结论、局限；每个单元格末尾也要标 [文件名 第X页]；"
            "资料里出现的每一篇论文都必须单独占一行，某篇没有某列的信息就写“未报告”，"
            "绝对不能因为信息少就省略这篇；不要编造。"
        )

    user = f"资料：\n{context}\n\n问题：{question}\n\n请回答，并标注页码。"
    return system, user


# ---------------- 引用校验：把“标了页码”变成“页码可验证” ----------------
#
# 之前版本只靠 Prompt 要求模型标页码，模型完全可以标一个不存在的页码。
# 这里做三件事：
#   1. 解析回答里的 [文件 第X页]；
#   2. 检查这一页是否真的在本次检索到的原文里（不在 = 疑似编造）；
#   3. 检查这句话和该页原文有没有词面支撑（太低 = 待人工核对）；
#   4. 找出该页最可能支撑这句话的原文句，给 UI 做“证据高亮”。

CITE = re.compile(r"[\[【]\s*([^\[\]【】\n]{1,120}?)\s*第\s*(\d{1,4})\s*页\s*[\]】]")
SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+|\n+")
BOUND = "。！？；\n|!?;"
REFUSAL_MARKS = (
    "没有足够证据", "资料中没有", "没有找到", "未找到", "无法回答",
    "未报告", "没有提及", "未提及", "资料不足", "不足以回答",
)


def _page_texts(chunks):
    """{(file, page): 该页所有 chunk 拼起来的原文}"""
    out = {}
    for c in chunks:
        out.setdefault((c.file, c.page), []).append(c.text)
    return {k: "\n".join(v) for k, v in out.items()}


def _resolve_file(raw: str, pages) -> str:
    """模型可能写全名、也可能写简称；能对上就用全名。"""
    raw = (raw or "").strip().strip("\"'“”")
    names = {f for f, _ in pages}
    if raw in names:
        return raw
    cands = [f for f in names if f.startswith(raw) or raw in f]
    if len(cands) == 1:
        return cands[0]
    return ""


def claim_before(text: str, pos: int, limit: int = 320) -> str:
    """取引用标记前面那句话，作为“待验证的结论”。"""
    start = max(0, pos - limit)
    for i in range(pos - 1, start - 1, -1):
        if text[i] in BOUND:
            start = i + 1
            break
    return text[start:pos].strip(" \t\r\n-•*#>")


def split_sentences(text: str):
    return [s.strip() for s in SENT_SPLIT.split(text or "") if s and s.strip()]


def _support(claim: str, page_text: str):
    """词面支撑度：结论里的实词有多大比例能在这一页找到。

    中文结论 + 英文论文时，只拿英文/数字 token 去比，
    否则中文二元组永远匹配不上，会把真引用误判成“弱”。
    两种文字没有可比 token 时返回 None：只做页码校验，不硬判弱。
    """
    q = tokenize(claim)
    if not q:
        return None
    if not CJK.search(page_text):
        q = {t for t in q if t.isascii()}
        if not q:
            return None
    d = tokenize(page_text)
    if not d:
        return None
    return len(q & d) / len(q)


def best_evidence(claim: str, page_text: str) -> str:
    """在这一页里找最像“支撑这句话”的原句，用于 UI 高亮。"""
    q = tokenize(claim)
    if not q:
        return ""
    if not CJK.search(page_text):
        q = {t for t in q if t.isascii()}
    if not q:
        return ""
    best, best_score = "", 0.0
    for sent in split_sentences(page_text):
        if len(sent) < 15:
            continue
        s = tokenize(sent)
        if not s:
            continue
        score = len(q & s) / len(q)
        if score > best_score:
            best, best_score = sent, score
    return best if best_score >= 0.3 else ""


def is_refusal(answer: str) -> bool:
    head = (answer or "")[:300]
    return not CITE.search(head) and any(m in head for m in REFUSAL_MARKS)


def verify_answer(answer: str, chunks, threshold: float = 0.30):
    """返回可序列化的校验报告，UI / 评测脚本共用。"""
    pages = _page_texts(chunks)
    items, seen = [], set()

    for m in CITE.finditer(answer or ""):
        raw, page = m.group(1), int(m.group(2))
        f = _resolve_file(raw, pages)
        claim = claim_before(answer, m.start())
        if (f, page, claim) in seen:
            continue
        seen.add((f, page, claim))

        base = {"file": f or raw.strip(), "page": page, "claim": claim,
                "raw": m.group(0), "support": 0.0, "evidence": ""}
        if not f or (f, page) not in pages:
            items.append({**base, "status": "missing"})
            continue
        text = pages[(f, page)]
        support = _support(claim, text)
        items.append({
            **base,
            "status": "ok" if support is None or support >= threshold else "weak",
            "support": None if support is None else round(support, 2),
            "evidence": best_evidence(claim, text),
        })

    ok = sum(1 for i in items if i["status"] == "ok")
    weak = sum(1 for i in items if i["status"] == "weak")
    missing = sum(1 for i in items if i["status"] == "missing")
    return {
        "citations": items,
        "total": len(items),
        "ok": ok,
        "weak": weak,
        "missing": missing,
        "pass": bool(items) and missing == 0 and weak == 0,
        "refused": is_refusal(answer),
    }


def ask_deepseek_fix(api_key: str, question: str, chunks, answer: str,
                     report: dict, model: str = "deepseek-chat") -> str:
    """只重写没通过校验的句子；通过的原样保留。最多用一次，避免死循环。"""
    bad = [i for i in report.get("citations", []) if i["status"] != "ok"]
    if not bad:
        return answer
    lines = [
        f"{n}. 原句：{i['claim'] or '（空）'}\n   引用：{i['raw']}"
        for n, i in enumerate(bad, start=1)
    ]
    system = (
        "你是论文阅读助手的校对器。下面这份回答里，有几处引用的页码在检索到的原文中"
        "找不到，或者原文支撑不足。请只重写这些句子：能由原文支撑的，改成更贴近原文的表述"
        "并标注正确的 [文件名 第X页]；不能支撑的，改写为“资料中没有足够证据”。"
        "其余句子一字不改，最后输出完整修订稿，不要解释。"
    )
    user = (
        f"问题：{question}\n\n需要修订的地方：\n" + "\n".join(lines) +
        f"\n\n原回答：\n{answer}\n\n请输出完整修订稿。"
    )
    return _chat(api_key, system, user, model=model, temperature=0.1, max_tokens=1600)


def _chat(api_key: str, system: str, user: str, model: str = "deepseek-chat",
          temperature: float = 0.1, max_tokens: int = 1600) -> str:
    import requests

    if not api_key:
        raise ValueError("还没有填 DeepSeek API Key。")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers=headers,
        json=payload,
        timeout=90,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API 返回 {resp.status_code}：{resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()


def ask_deepseek(api_key: str, question: str, chunks, model: str = "deepseek-chat",
                 compare: bool = False) -> str:
    system, user = build_prompt(question, chunks, compare=compare)
    return _chat(api_key, system, user, model=model, temperature=0.1, max_tokens=1200)


def translate_text(api_key: str, text: str, target: str = "中文",
                   model: str = "deepseek-chat") -> str:
    system = (
        "你是学术论文翻译助手。忠实翻译，不增删内容，不解释。"
        "保留公式、代码、术语、引用编号和章节编号；专有名词首次出现时可在括号里保留英文。"
        "只输出译文，不要加“以下是翻译”。"
    )
    user = f"目标语言：{target}\n\n原文：\n{text[:5000]}"
    return _chat(api_key, system, user, model=model, temperature=0.2, max_tokens=2200)


def write_discussion(api_key: str, topic: str, chunks, style: str = "Related Work",
                     model: str = "deepseek-chat") -> str:
    context = "\n\n".join(
        f"【资料{i}】文件：{c.file}  第{c.page}页\n{c.text[:1200]}"
        for i, c in enumerate(chunks, start=1)
    )
    system = (
        "你是软件测试方向的论文写作助手。只使用给定资料，不要编造文献、数字或结论。"
        "写 300–500 字中文论述，按主题组织，不要写成逐篇摘要。"
        "每一个事实句末尾标 [文件名 第X页]；资料不足时明确说“资料不足”，不要补外部知识。"
    )
    user = (
        f"写作类型：{style}\n主题：{topic}\n\n资料：\n{context}\n\n"
        "请直接输出论述正文，使用 Markdown。"
    )
    return _chat(api_key, system, user, model=model, temperature=0.3, max_tokens=1800)


# ---------------- 语义去重与版本差异（官方测试用例 3.1） ----------------
#
# 官方要求：按“内容”而不是文件名识别同一论文的不同版本（如 1706.03762v1 vs v7），
# 提示用户保留/覆盖，并能总结两版差异。这里不引入向量库：
# 用字符 5-gram 的 Jaccard 做指纹，配合 arXiv 版本号判断新旧。

ARXIV = re.compile(r"(\d{4}\.\d{4,5})(?:v(\d+))?", re.I)


def _fingerprint(pages, limit: int = 8000, n: int = 5):
    """字符 n-gram 的多重集指纹：重复句子多的文档也不会被算成“不像”。"""
    from collections import Counter

    text = re.sub(r"\s+", " ", " ".join(p.text for p in pages).lower())[:limit]
    if len(text) < 300:
        return Counter()
    return Counter(text[i:i + n] for i in range(0, len(text) - n, 2))


def _looks_same_paper(title_like: str, text: str):
    """先看 arXiv ID / 标题，能直接判定就不用算相似度。"""
    for source in (title_like, text[:3000]):
        m = ARXIV.search(source)
        if m:
            return m.group(1), int(m.group(2) or 0)
    return "", 0


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


def find_duplicate_groups(papers, threshold: float = 0.75):
    """papers: {文件名: [Page, ...]}。返回 [{files, keep, similarity, arxiv_id, versions}]。"""
    names = list(papers)
    fps = {f: _fingerprint(papers[f]) for f in names}
    ids = {}
    for f in names:
        text = " ".join(p.text for p in papers[f])
        ids[f] = _looks_same_paper(f, text)

    parent = {f: f for f in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    sims = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same_id = ids[a][0] and ids[a][0] == ids[b][0]
            sim = 1.0 if same_id else _jaccard(fps[a], fps[b])
            sims[frozenset((a, b))] = sim
            if sim >= threshold:
                union(a, b)

    groups = {}
    for f in names:
        groups.setdefault(find(f), []).append(f)

    out = []
    for files in groups.values():
        if len(files) < 2:
            continue
        files = sorted(files)
        keep = max(files, key=lambda f: (ids[f][1], sum(len(p.text) for p in papers[f]), f))
        pair_sims = [sims[frozenset((a, b))]
                     for i, a in enumerate(files) for b in files[i + 1:]]
        out.append({
            "files": files,
            "keep": keep,
            "similarity": round(max(pair_sims), 3),
            "arxiv_id": ids[keep][0],
            "versions": {f: ids[f][1] for f in files},
        })
    return out


def diff_versions(old_pages, new_pages, max_examples: int = 5):
    """按句子粒度对比两个版本，带页码锚点。返回可直接展示的 dict。"""
    import difflib

    def sents(pages):
        out = []
        for p in pages:
            for s in split_sentences(p.text):
                s = s.strip()
                if len(s) >= 25:
                    out.append((p.page, s))
        return out

    old, new = sents(old_pages), sents(new_pages)
    sm = difflib.SequenceMatcher(a=[s for _, s in old], b=[s for _, s in new], autojunk=False)

    added, removed, changed = [], [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            added += [(new[j][0], new[j][1]) for j in range(j1, j2)]
        elif tag == "delete":
            removed += [(old[i][0], old[i][1]) for i in range(i1, i2)]
        elif tag == "replace":
            changed.append({"old_page": old[i1][0], "new_page": new[j1][0],
                            "old": old[i1][1], "new": new[j1][1]})
            added += [(new[j][0], new[j][1]) for j in range(j1, j2)]
            removed += [(old[i][0], old[i][1]) for i in range(i1, i2)]

    pages_touched = sorted({p for p, _ in added + removed} |
                           {c["new_page"] for c in changed})
    return {
        "added": added[:max_examples],
        "removed": removed[:max_examples],
        "changed": changed[:max_examples],
        "counts": {"added": len(added), "removed": len(removed), "changed": len(changed)},
        "pages": pages_touched[:8],
    }


def format_diff(diff, old_name: str, new_name: str) -> str:
    c = diff["counts"]
    lines = [f"**{old_name} → {new_name}**：新增 {c['added']} 句、删除 {c['removed']} 句、"
             f"改写 {c['changed']} 处；变化集中在第 {', '.join(map(str, diff['pages']))} 页左右。", ""]
    for label, items in (("新增", diff["added"]), ("删除", diff["removed"])):
        if items:
            lines.append(f"**{label}（示例）**")
            lines += [f"- 第 {pg} 页：{s[:90]}" for pg, s in items[:3]]
            lines.append("")
    if diff["changed"]:
        lines.append("**改写（示例）**")
        for ch in diff["changed"][:3]:
            lines.append(f"- 第 {ch['old_page']} 页 → 第 {ch['new_page']} 页：{ch['new'][:80]}")
    return "\n".join(lines).strip()


# ---------------- arXiv 检索入库（工具调用 / Function Calling） ----------------

ARXIV_NS = {"a": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_CACHE = {}
GITHUB_URL = re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def parse_arxiv_feed(xml_text: str):
    """解析 arXiv Atom 返回；拆出来，离线也能单测。"""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", ARXIV_NS):
        def text(tag):
            node = e.find(tag, ARXIV_NS)
            return (node.text or "").strip() if node is not None else ""

        pdf_url = ""
        for link in e.findall("a:link", ARXIV_NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")

        summary = " ".join(text("a:summary").split())
        comment = text("arxiv:comment")
        gh = GITHUB_URL.search(summary + " " + comment)
        out.append({
            "id": text("a:id").rsplit("/", 1)[-1],
            "title": " ".join(text("a:title").split()),
            "authors": ", ".join(
                a.findtext("a:name", default="", namespaces=ARXIV_NS)
                for a in e.findall("a:author", ARXIV_NS)
            ),
            "published": text("a:published")[:10],
            "summary": summary,
            "pdf_url": pdf_url,
            "github": gh.group(0) if gh else "",
        })
    return out


def arxiv_search(query: str, max_results: int = 5, timeout: int = 45, retries: int = 3):
    """关键词检索或按 ID 精确取；这是“Agent 调用外部 API”的最小可用实现。"""
    import time as _time

    import requests

    q = (query or "").strip()
    if not q:
        raise ValueError("先输入关键词或 arXiv ID。")

    # 同一进程内缓存 10 分钟：arXiv 对重复请求会限流（429），
    # 演示时评委多点两次“搜索”也不至于被掐。
    now = _time.monotonic()
    cached = _ARXIV_CACHE.get((q.lower(), int(max_results)))
    if cached and now - cached[0] < 600:
        return cached[1]

    params = {"max_results": max(1, min(int(max_results), 20))}
    if ARXIV_ID.match(q):
        params["id_list"] = q
    else:
        params["search_query"] = f"all:{q}"
        params["sortBy"] = "relevance"

    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get("https://export.arxiv.org/api/query", params=params,
                                timeout=(10, timeout))
            if resp.status_code == 429:
                raise RuntimeError("arXiv 限流（HTTP 429），等 30 秒再试")
            if resp.status_code != 200:
                raise RuntimeError(f"arXiv 返回 HTTP {resp.status_code}")
            items = parse_arxiv_feed(resp.text)
            _ARXIV_CACHE[(q.lower(), int(max_results))] = (now, items)
            return items
        except Exception as exc:  # 网络抖动常见，重试一次再报错
            last_err = exc
            if attempt + 1 < retries:
                _time.sleep(6)
    raise RuntimeError(f"arXiv 暂时无响应（已重试 {retries} 次）：{last_err}")


def download_arxiv_pdf(pdf_url: str, timeout: int = 40) -> bytes:
    import requests

    if not pdf_url:
        raise ValueError("这条记录没有 PDF 链接。")
    resp = requests.get(pdf_url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"下载失败：HTTP {resp.status_code}")
    return resp.content


# ---------------- arXiv 最近使用（最多 5 条） ----------------

HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arxiv_history.json")


def load_history(limit: int = 5):
    """读“最近使用过的 arXiv 论文”。文件不存在或损坏都返回空列表。"""
    try:
        with open(HISTORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data[:limit] if isinstance(data, list) else []
    except Exception:
        return []


def record_history(item, limit: int = 5):
    """记一条使用记录：同 ID 去重、新的在前、最多 limit 条。写不进去也不报错。"""
    if not item or not item.get("id"):
        return load_history(limit)
    entry = {k: item.get(k, "") for k in
             ("id", "title", "authors", "published", "pdf_url", "github")}
    entry["used_at"] = time.strftime("%Y-%m-%d %H:%M")
    data = [h for h in load_history(50) if h.get("id") != entry["id"]]
    data.insert(0, entry)
    data = data[:limit]
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except Exception:
        pass
    return data


# ---------------- Agent 规划：从文献库自己找相关论文 ----------------

STOP_EN = set(
    "the and for with that this these those from into than then they their there "
    "have has had are was were will would can could may might not but also such "
    "paper papers approach method methods result results show shows propose proposed "
    "using used use based on in of to a an by as at is are be our we it its".split()
)


def extract_topic_terms(chunks, limit: int = 8):
    """从库里论文的高频英文词里抽主题词，作为检索种子。"""
    from collections import Counter

    cnt = Counter()
    for c in chunks:
        for w, n in token_counts(c.text).items():
            if w.isascii() and w.isalpha() and len(w) >= 4 and w not in STOP_EN:
                cnt[w] += n
    return [w for w, _ in cnt.most_common(limit)]


def _library_ids(chunks):
    """把已入库文件名还原成 arXiv ID（2305.04764v2.pdf -> 2305.04764）。"""
    ids = set()
    for c in chunks:
        name = c.file.rsplit(".", 1)[0]
        ids.add(name.split("v")[0] if re.match(r"^\d{4}\.\d{4,5}", name) else name.lower())
    return ids


def rank_candidates(items, existing_ids, terms, limit: int = 5):
    """过滤掉已在库的、按标题/摘要命中主题词的数量排序。"""
    out = []
    for it in items:
        aid = it["id"].split("v")[0]
        if aid in existing_ids:
            continue
        hay = (it["title"] + " " + it.get("summary", "")).lower()
        hits = [t for t in terms if t in hay]
        if not hits:
            continue
        out.append({**it, "hits": hits[:4], "score": len(hits)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:limit]


def plan_related(chunks, max_results: int = 10, limit: int = 5):
    """一次“自主规划”：抽主题词 → 调 arXiv → 过滤 → 排序，并返回每一步。"""
    terms = extract_topic_terms(chunks, limit=8)
    if not terms:
        return {"steps": ["库里还没有可用的英文文本，先上传论文"], "items": [], "terms": []}

    top = terms[:3]
    if len(top) >= 2:
        query = f"all:{top[0]} AND all:{top[1]}"
    else:
        query = f"all:{top[0]}"

    steps = [
        f"读取库内论文，统计高频术语，选出：{'、'.join(top)}",
        f"构造检索式：{query}",
    ]
    try:
        raw = arxiv_search(query, max_results=max_results)
    except Exception as exc:
        steps.append(f"调用 arXiv 失败：{exc}")
        return {"steps": steps, "items": [], "terms": terms}

    steps.append(f"arXiv 返回 {len(raw)} 条")
    existing = _library_ids(chunks)
    fresh = [r for r in raw if r["id"].split("v")[0] not in existing]
    items = rank_candidates(raw, existing, terms, limit=limit)
    steps.append(f"剔除已在库 {len(raw) - len(fresh)} 条、主题不相关 {max(0, len(fresh) - len(items))} 条，"
                 f"推荐 {len(items)} 条")
    return {"steps": steps, "items": items, "terms": terms, "query": query}


# ---------------- 流式输出（SSE） ----------------

def iter_sse_content(lines):
    """从 DeepSeek 的 SSE 行里取增量文本。拆出来，不联网也能测。"""
    import json

    for line in lines:
        line = (line or "").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = (choices[0].get("delta") or {}).get("content") or ""
        if delta:
            yield delta


def ask_deepseek_stream(api_key: str, question: str, chunks, compare: bool = False,
                        model: str = "deepseek-chat"):
    """流式版问答：边生成边显示，UI 用 st.write_stream 消费。"""
    import json

    import requests

    if not api_key:
        raise ValueError("还没有填 DeepSeek API Key。")
    system, user = build_prompt(question, chunks, compare=compare)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.1,
        "max_tokens": 1200,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {api_key.strip()}",
               "Content-Type": "application/json"}
    with requests.post("https://api.deepseek.com/chat/completions", headers=headers,
                       json=payload, timeout=120, stream=True) as resp:
        if resp.status_code != 200:
            body = resp.text[:200]
            raise RuntimeError(f"API 返回 {resp.status_code}：{body}")
        yield from iter_sse_content(resp.iter_lines(decode_unicode=True))


def render_page_png(pdf_bytes: bytes, page: int, dpi: int = 110) -> bytes:
    """把 PDF 某一页渲染成 PNG，用于“左原文版式 / 右译文”的联动阅读。

    不做前端 PDF.js，直接用 PyMuPDF 出图：省依赖、部署稳，评委看到的就是原版式。
    """
    if not pdf_bytes:
        return b""
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page < 1 or page > doc.page_count:
            return b""
        return doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()


def extract_references(pages, max_refs: int = 40):
    """从每篇论文的 References 段落抽参考文献；抽不到就返回空。"""
    refs = []
    head = re.compile(r"(?i)^\s*(references|bibliography|参考文献)\s*$")
    entry = re.compile(r"^\s*(\[\d+\]|\d+\.)\s+")

    for p in pages:
        lines = p.text.splitlines()
        start = None
        for i, line in enumerate(lines):
            if head.match(line.strip()):
                start = i + 1
        if start is None:
            continue
        for line in lines[start:]:
            line = clean_text(line)
            if len(line) < 20:
                continue
            if entry.match(line) or re.match(r"^[A-Z][A-Za-z\-]+,\s", line):
                refs.append((p.file, p.page, line))

    if not refs:
        for p in pages:
            for line in p.text.splitlines():
                line = clean_text(line)
                if re.match(r"^\[\d+\]\s+", line) and len(line) >= 20:
                    refs.append((p.file, p.page, line))

    seen, out = set(), []
    for item in refs:
        key = item[2][:80]
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out[:max_refs]
