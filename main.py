#!/usr/bin/env python3
"""
政策风向与体制逻辑追踪 — 每周日 7:00 AM 推送
抓取国务院/发改委/财政部等官方政策文件 + 外部智库分析
DeepSeek 解读体制内文本语言与政策逻辑
"""

import os
import sys
import re
import json
import smtplib
import logging
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import feedparser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json")

# ─── 数据源 ──────────────────────────────────────────────────────────────────

# 中国官方政策（网页抓取）
GOV_SOURCES = {
    "国务院政策": "https://www.gov.cn/zhengce/zuixin.htm",
    "发改委政策": "https://www.ndrc.gov.cn/xwdt/xwfb/",
    "财政部政策": "https://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/",
}

# 外部深度分析（RSS）
EXTERNAL_RSS = {
    "外交学者": "https://thediplomat.com/feed/",
    "南华早报": "https://www.scmp.com/rss/4/feed",
    "第六声": "https://www.sixthtone.com/rss",
    "LSE中国": "https://blogs.lse.ac.uk/cff/feed/",
}

MAX_PER_SOURCE = 3
DAILY_TOTAL = 8

# ─── 网页抓取 ────────────────────────────────────────────────────────────────


def get_session():
    s = requests.Session()
    proxy_url = os.environ.get("HTTPS_PROXY", os.environ.get("https_proxy", ""))
    if proxy_url:
        s.proxies = {"https": proxy_url, "http": proxy_url.replace("https", "http")}
    s.trust_env = False
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; PolicyTracker/1.0)"})
    return s


def scrape_gov_page(name: str, url: str, max_n: int) -> List[Dict[str, Any]]:
    """抓取中国政府网页的政策文件列表。"""
    docs: List[Dict[str, Any]] = []
    try:
        session = get_session()
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        # 处理中文编码
        resp.encoding = resp.apparent_encoding or 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')

        # 找政策文件链接（常见 class/id 模式）
        items = (
            soup.select('ul.listTxt li') or
            soup.select('.news_box li') or
            soup.select('.list li') or
            soup.select('li a[href*="content"]') or
            soup.select('a[href*="zhengce"]') or
            soup.find_all('a', href=re.compile(r'(content|zhengce|xinxi|fabu)'))
        )

        count = 0
        seen = set()
        for item in items:
            if count >= max_n:
                break
            a_tag = item if item.name == 'a' else item.find('a')
            if not a_tag:
                continue
            title = a_tag.get_text(strip=True)
            href = a_tag.get('href', '')
            if not title or not href or len(title) < 8:
                continue
            # 补全 URL
            if href.startswith('/'):
                from urllib.parse import urljoin
                href = urljoin(url, href)
            if href in seen:
                continue
            seen.add(href)

            # 日期
            date_span = item.find('span') or item.find('em')
            pub_date = date_span.get_text(strip=True) if date_span else ''

            # 抓取全文
            full_text = fetch_full_text(href)
            abstract = full_text[:3000] if full_text else title

            docs.append({
                "title": title, "abstract": abstract, "url": href,
                "published": pub_date or datetime.now().strftime("%Y-%m-%d"),
                "source": name, "type": "official",
            })
            count += 1

        logger.info("  %s: %d 篇", name, len(docs))
    except Exception as exc:
        logger.warning("  %s 抓取失败: %s", name, exc)
    return docs


def fetch_full_text(url: str) -> str:
    """抓取文章/文件全文。"""
    try:
        session = get_session()
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'img']):
            tag.decompose()
        # 政府文件主体通常在 .article, .content, #content, .TRS_Editor 等
        body = (
            soup.find('div', class_=re.compile(r'article|content|TRS|main|text', re.I)) or
            soup.find('article') or
            soup.find('main') or
            soup.body
        )
        if body:
            text = body.get_text(separator=' ', strip=True)
            return re.sub(r'\s+', ' ', text)[:5000]
    except Exception:
        pass
    return ""


def fetch_rss(name: str, url: str, max_n: int) -> List[Dict[str, Any]]:
    """RSS 获取外部分析。"""
    docs: List[Dict[str, Any]] = []
    try:
        session = get_session()
        resp = session.get(url, timeout=45)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        # 过滤中国/政策相关内容
        china_kw = re.compile(
            r'china|中国|beijing|xi|communist|politburo|policy|regulation|reform|economy|trade|belt.*road|一带一路|双循环|共同富裕',
            re.IGNORECASE,
        )
        for entry in feed.entries[:max_n * 2]:
            title = entry.get("title", "")
            summary = entry.get("summary", entry.get("description", ""))
            summary = re.sub(r"<[^>]+>", "", summary)
            summary = " ".join(summary.split())
            if not china_kw.search(f"{title} {summary}"):
                continue
            pub_str = entry.get("published", entry.get("updated", ""))
            pub_date = pub_str[:10] if pub_str else ""
            link = entry.get("link", "")
            full_text = fetch_full_text(link)
            content = full_text if len(full_text) > len(summary) * 2 else summary
            docs.append({
                "title": title, "abstract": content, "url": link,
                "published": pub_date or datetime.now().strftime("%Y-%m-%d"),
                "source": name, "type": "external",
            })
            if len(docs) >= max_n:
                break
        logger.info("  %s: %d 篇", name, len(docs))
    except Exception as exc:
        logger.warning("  %s 失败: %s", name, exc)
    return docs


def dedup(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for d in docs:
        key = d["title"][:60]
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


# ─── DeepSeek 政策解读 ───────────────────────────────────────────────────────


def deepseek_analyze(doc: Dict[str, Any], api_key: str) -> Optional[str]:
    if not api_key:
        return None

    doc_type = "官方政策文件" if doc["type"] == "official" else "外部政策分析"
    prompt = f"""你是一位资深政策分析师，擅长解读中国体制内文本逻辑与政策信号。

来源：{doc['source']}（{doc_type}）
标题：{doc['title']}
正文：{doc['abstract'][:4000]}

请按以下三段式输出（每段 4-6 句，专业但不官僚，纯文本不要Markdown）：

【政策要点】
这篇文件/文章的核心内容是什么？涉及哪些具体领域（产业、财税、金融、民生、外贸等）？有哪些关键的新提法、新表述或量化目标？请用通俗语言概括。

【体制逻辑】
从制度设计角度看，这个政策背后的驱动逻辑是什么？它反映了决策层对什么问题的判断？和近期的其他政策有什么关联？文件中那些看似套话的表述（如"稳中求进""统筹兼顾"）在当下语境中实际传递了什么信号？

【影响推演】
这个政策对不同主体（地方政府、企业、投资者、普通民众）可能产生什么影响？哪些行业或领域会受益/承压？有什么值得持续跟踪的后续动向？"""

    try:
        resp = requests.post(
            DEEPSEEK_API,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": "你是资深中国政策分析师。请用中文解读政策文件，深入剖析体制内文本逻辑与政策信号。不要用Markdown，纯文本段落输出。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
                "max_tokens": 1000,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("DeepSeek 失败 (%s…): %s", doc['title'][:30], exc)
        return None


def batch_analyze(docs: List[Dict[str, Any]], api_key: str, max_n: int) -> None:
    targets = docs[:max_n]
    logger.info("DeepSeek: 解读 %d 篇 …", len(targets))

    def _one(d):
        return docs.index(d), deepseek_analyze(d, api_key)

    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed({ex.submit(_one, d): d for d in targets}):
            try:
                idx, result = fut.result()
                if result:
                    docs[idx]["interpretation"] = result
            except Exception as exc:
                logger.warning("  异常: %s", exc)
    logger.info("  ✅ %d/%d 完成", sum(1 for d in targets if d.get("interpretation")), len(targets))


# ─── HTML ────────────────────────────────────────────────────────────────────


def build_html(docs: List[Dict[str, Any]], date_str: str, week_num: int) -> str:
    interpreted = [d for d in docs if d.get("interpretation")]
    if not interpreted:
        return "<p>暂无解读内容</p>"

    type_emoji = {"official": "🏛️", "external": "🌐"}
    type_label = {"official": "官方文件", "external": "外部分析"}
    type_color = {"official": "#dc2626", "external": "#2563eb"}

    cards = []
    for doc in interpreted:
        dt = doc.get("type", "external")
        emoji = type_emoji.get(dt, "📄")
        label = type_label.get(dt, "")
        color = type_color.get(dt, "#6b7280")
        interp = doc.get("interpretation", "")

        sections: Dict[str, str] = {}
        for key in ["政策要点", "体制逻辑", "影响推演"]:
            marker = f"【{key}】"
            if marker in interp:
                parts = interp.split(marker, 1)
                if len(parts) > 1:
                    sections[key] = parts[1].split("【")[0].strip()

        sec_config = [
            ("政策要点", "#dc2626", "1"),
            ("体制逻辑", "#7c3aed", "2"),
            ("影响推演", "#0891b2", "3"),
        ]
        interp_html = ""
        for key, sc, num in sec_config:
            text = sections.get(key, "")
            if text:
                interp_html += f"""
    <div style="margin-top:14px;background-color:#f8fafc;border-left:3px solid {sc};border-radius:0 8px 8px 0;padding:14px 18px;">
    <span style="font-size:16px;font-weight:800;color:{sc};">{num}. {key}</span>
    <div style="font-size:16px;color:#374151;line-height:1.9;margin-top:6px;">{text}</div>
    </div>"""

        cards.append(f"""
<tr>
<td style="padding:24px 32px;border-bottom:1px solid #e5e7eb;">
<div style="margin-bottom:8px;">
<span style="font-size:12px;padding:4px 12px;border-radius:6px;background-color:{color}12;color:{color};font-weight:700;">{emoji} {doc['source']} · {label}</span>
</div>
<div style="font-size:20px;font-weight:800;color:#111827;line-height:1.4;margin-bottom:12px;">{doc['title']}</div>
<div style="font-size:13px;color:#6b7280;margin-bottom:16px;">{doc.get('published', '')} &nbsp;<a href="{doc['url']}" style="color:#2563eb;font-weight:600;text-decoration:none;">阅读全文 →</a></div>
{interp_html}
</td>
</tr>""")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>政策风向与体制逻辑追踪</title>
</head>
<body style="margin:0;padding:0;background-color:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei','Helvetica Neue',sans-serif;">

<table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f1f5f9;padding:20px 0;">
<tr><td align="center">
<table width="920" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:12px;overflow:hidden;max-width:920px;box-shadow:0 1px 3px rgba(0,0,0,0.08);">

<tr>
<td style="background:linear-gradient(135deg,#991b1b 0%,#7f1d1d 100%);padding:40px 36px;text-align:center;">
<div style="font-size:14px;letter-spacing:3px;color:rgba(255,255,255,0.7);margin-bottom:8px;">第 {week_num} 期 · {date_str}</div>
<h1 style="margin:0;font-size:30px;font-weight:800;color:#ffffff;">📜 政策风向与体制逻辑追踪</h1>
<div style="margin-top:10px;font-size:15px;color:rgba(255,255,255,0.65);">追踪官方文件 · 解读体制语言 · 推演政策影响</div>
</td>
</tr>

<tr>
<td style="padding:20px 32px 0;">
<div style="background-color:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:14px 18px;font-size:15px;color:#991b1b;line-height:1.7;">
📌 本周政策文件精选，每篇附 <b style="color:#dc2626;">DeepSeek 三段式解读</b>（政策要点 → 体制逻辑 → 影响推演），不看新闻通稿，直接读懂文件背后的治理逻辑。
</div>
</td>
</tr>

{''.join(cards)}

<tr>
<td style="background-color:#f8fafc;padding:24px 32px;text-align:center;border-top:1px solid #e5e7eb;">
<div style="font-size:13px;color:#9ca3af;line-height:1.8;">
政策风向与体制逻辑追踪 &bull; 第 {week_num} 期 &bull; {date_str}<br>
每周日 7:00 AM 自动推送 &bull; 来源：国务院/发改委/财政部 + 外部智库<br>
AI 解读由 DeepSeek 生成 &bull; 仅供政策学习参考
</div>
</td>
</tr>

</table>
</td></tr>
</table>
</body>
</html>"""


# ─── 历史与邮件 ──────────────────────────────────────────────────────────────

def load_history() -> Dict[str, Any]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {"week": 0}


def save_history(h: Dict[str, Any]) -> None:
    with open(HISTORY_FILE, "w") as f:
        json.dump(h, f, ensure_ascii=False, indent=2)


def send_email(html: str, date_str: str, week_num: int) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = Header(f"📜 政策风向追踪 — 第{week_num}期 {date_str}", "utf-8")
    msg["From"] = f"Policy Tracker <{os.environ['SENDER_EMAIL']}>"
    msg["To"] = os.environ["RECIPIENT_EMAIL"]
    msg.attach(MIMEText(html, "html", "utf-8"))
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(os.environ["SENDER_EMAIL"], os.environ["SENDER_PASSWORD"])
        server.send_message(msg)
    logger.info("邮件发送成功！")


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    try:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info("=== 政策风向追踪 %s ===", date_str)

        history = load_history()

        all_docs: List[Dict[str, Any]] = []

        # 1. 官方政策文件
        for name, url in GOV_SOURCES.items():
            logger.info("抓取: %s", name)
            docs = scrape_gov_page(name, url, MAX_PER_SOURCE)
            all_docs.extend(docs)

        # 2. 外部分析
        for name, url in EXTERNAL_RSS.items():
            logger.info("RSS: %s", name)
            docs = fetch_rss(name, url, MAX_PER_SOURCE)
            all_docs.extend(docs)

        all_docs = dedup(all_docs)
        logger.info("总计 %d 篇（去重后）", len(all_docs))

        if not all_docs:
            logger.warning("无内容，跳过")
            return

        # 3. DeepSeek 解读
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if api_key:
            batch_analyze(all_docs, api_key, DAILY_TOTAL)
        else:
            logger.error("未设置 DEEPSEEK_API_KEY")
            sys.exit(1)

        # 4. 更新历史
        history["week"] = history.get("week", 0) + 1
        save_history(history)
        week_num = history["week"]

        # 5. HTML + 发送
        html = build_html(all_docs, date_str, week_num)
        logger.info("HTML: %.1f KB, 解读 %d 篇", len(html) / 1024, sum(1 for d in all_docs if d.get("interpretation")))
        send_email(html, date_str, week_num)
        logger.info("完成！第 %d 期", week_num)
    except Exception as exc:
        logger.error("致命: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
