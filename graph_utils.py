# -*- coding: utf-8 -*-
"""Build a lightweight citation map as inline SVG (no extra dependencies)."""

import html


def _truncate(text: str, n: int = 56) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def build_citation_svg(papers, refs, width: int = 1100):
    """papers: list[str]; refs: list[(file, page, text)]."""
    papers = list(dict.fromkeys(papers))
    refs = refs[:18]
    height = max(460, 130 + len(refs) * 46)

    left_x, left_w = 30, 230
    right_x, right_w = 400, 650
    top = 70

    # paper y positions
    paper_y = {}
    if len(papers) == 1:
        paper_y[papers[0]] = 230
    else:
        step = min(150, max(90, (height - 160) / max(1, len(papers) - 1)))
        for i, p in enumerate(papers):
            paper_y[p] = 100 + i * step

    ref_y = [top + i * 46 for i in range(len(refs))]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {height}" '
        f'font-family="Microsoft YaHei, Noto Sans SC, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#FBF9F2"/>',
        f'<text x="30" y="32" font-size="18" font-weight="700" fill="#14201C">'
        f'论文 → 参考文献（轻量版）</text>',
        f'<text x="30" y="54" font-size="11" fill="#6F7C76">'
        f'只解析 References 列表，不做跨库引用网络</text>',
    ]

    # edges first
    for i, (f, page, text) in enumerate(refs):
        if f not in paper_y:
            continue
        y1 = paper_y[f]
        y2 = ref_y[i]
        parts.append(
            f'<path d="M {left_x + left_w} {y1} C {left_x + left_w + 70} {y1}, '
            f'{right_x - 70} {y2}, {right_x} {y2}" fill="none" '
            f'stroke="#2BD3AE" stroke-width="1.4" opacity="0.55"/>'
        )

    # paper nodes
    for p in papers:
        y = paper_y[p]
        parts.append(
            f'<rect x="{left_x}" y="{y - 24}" width="{left_w}" height="48" rx="9" '
            f'fill="#0D211B" stroke="#2E5B4B"/>'
        )
        parts.append(
            f'<text x="{left_x + 12}" y="{y - 2}" font-size="11" fill="#9FB8AE">论文</text>'
        )
        parts.append(
            f'<text x="{left_x + 12}" y="{y + 15}" font-size="12" font-weight="700" fill="#FFFFFF">'
            f'{_esc(_truncate(p, 18))}</text>'
        )

    # ref nodes
    for i, (f, page, text) in enumerate(refs):
        y = ref_y[i]
        parts.append(
            f'<rect x="{right_x}" y="{y - 17}" width="{right_w}" height="34" rx="7" '
            f'fill="#FFFFFF" stroke="#D9D2C1"/>'
        )
        parts.append(
            f'<text x="{right_x + 10}" y="{y - 2}" font-size="10" fill="#0E7C66">'
            f'{_esc(f)} · 第{page}页</text>'
        )
        parts.append(
            f'<text x="{right_x + 10}" y="{y + 13}" font-size="11" fill="#14201C">'
            f'{_esc(_truncate(text, 50))}</text>'
        )

    parts.append(
        f'<text x="30" y="{height - 22}" font-size="10" fill="#6F7C76">'
        f'共 {len(papers)} 篇论文 / {len(refs)} 条参考文献；完整的跨论文引用图谱在下一步实现。</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts), height
