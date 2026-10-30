# -*- coding: utf-8 -*-
"""TestPaper Copilot 保命版：PDF 读取、切块、检索、带页码回答。"""

from dataclasses import dataclass
import re


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


def tokenize(text: str):
    text = (text or "").lower()
    tokens = set()

    for w in WORD.findall(text):
        if len(w) >= 2 and w not in STOP:
            tokens.add(w)

    chars = [c for c in text if CJK.match(c)]
    for i in range(len(chars) - 1):
        bigram = chars[i] + chars[i + 1]
        if bigram not in STOP:
            tokens.add(bigram)

    return tokens


def score_chunk(q_tokens, question: str, chunk: Chunk) -> float:
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


def search(question: str, chunks, top_k: int = 6):
    q_tokens = tokenize(question)
    scored = [(score_chunk(q_tokens, question, c), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [c for s, c in scored[:top_k] if s > 0]
    top_score = scored[0][0] if scored else 0.0
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
            "论文没写的写“未报告”，不要编造。"
        )

    user = f"资料：\n{context}\n\n问题：{question}\n\n请回答，并标注页码。"
    return system, user


def ask_deepseek(api_key: str, question: str, chunks, model: str = "deepseek-chat",
                 compare: bool = False) -> str:
    import requests

    if not api_key:
        raise ValueError("还没有填 DeepSeek API Key。")

    system, user = build_prompt(question, chunks, compare=compare)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": 1200,
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

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
