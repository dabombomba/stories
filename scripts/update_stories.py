#!/usr/bin/env python3
import json
import os
import re
import ssl
import time
import urllib.request

FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.cnn.com/rss/edition.rss",
    "https://feeds.npr.org/1001/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://feeds.reuters.com/reuters/topNews",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
]

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
]

ITEM_RE = re.compile(
    r"<item>[\s\S]*?<title>(.*?)</title>[\s\S]*?<link>(.*?)</link>[\s\S]*?<description>(.*?)</description>",
    re.I,
)

CTX = ssl.create_default_context()


def fetch(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "stories-updater/1.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as res:
        return res.read().decode("utf-8", "replace")


def clean(text):
    text = re.sub(r"<!\[CDATA\[|\]\]>", "", text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_articles():
    seen = set()
    articles = []
    for feed in FEEDS:
        try:
            xml = fetch(feed)
        except Exception as exc:
            print(f"feed fail {feed}: {exc}")
            continue
        for title, link, desc in ITEM_RE.findall(xml):
            title = clean(title)
            link = clean(link).split("?")[0]
            desc = clean(desc)[:180]
            if not title or not link.startswith("http"):
                continue
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)
            articles.append({"title": title, "link": link, "desc": desc})
    return articles


def gemini(api_key, prompt):
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1.05,
            "maxOutputTokens": 2048,
            "topP": 0.95,
        },
    }).encode()
    last_err = None
    for model in MODELS:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            raw = urllib.request.urlopen(req, timeout=60, context=CTX).read()
            data = json.loads(raw.decode())
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            if text.strip():
                return text
        except Exception as exc:
            last_err = exc
            print(f"{model} failed: {exc}")
            time.sleep(1)
    raise RuntimeError(f"all models failed: {last_err}")


def normalize_line(line, fallback_url):
    line = re.sub(r"\s+", " ", line).strip().strip("`").strip('"').strip("'")
    if not line:
        return None
    if not line.startswith("MrDestructoid"):
        line = "MrDestructoid " + line.lstrip()
    line = re.sub(r"\s*VoHiYo.*$", "", line, flags=re.I).strip()
    prefix = "MrDestructoid"
    story = line[len(prefix):].strip().upper()
    url = fallback_url
    m = re.search(r"(https?://[^\s]+)", line)
    if m:
        url = m.group(1)
        story = re.sub(r"https?://\S+", "", story).strip(" -")
    line = f"{prefix} {story} VoHiYo {url}"
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) > 400:
        extra = len(line) - 400
        keep = max(20, len(story) - extra - 1)
        story = story[:keep].rstrip()
        line = f"{prefix} {story} VoHiYo {url}"
        if len(line) > 400:
            return None
    if not line.startswith("MrDestructoid "):
        return None
    if " VoHiYo http" not in line:
        return None
    return line


def batch_prompt(batch):
    numbered = []
    for i, art in enumerate(batch, 1):
        numbered.append(
            f"{i}. Title: {art['title']}\n"
            f"   Summary: {art['desc']}\n"
            f"   URL: {art['link']}"
        )
    n = len(batch)
    return f"""Rewrite each news item into one Mad Libs line.

{chr(10).join(numbered)}

For every item:
- Keep original numbers and dates.
- Substitute ordinary verbs with violent or horror-tinged verbs of the same grammatical form.
- Replace nouns (including place names) with silly cartoonish nouns from mixed categories (utensils, candy, toys, creatures, sports gear, foods). Keep the news followable.
- Do not reuse the same silly noun or horror verb across lines.
- Select a different category/region feel for each line.

Output exactly {n} lines, one per item, in the same order.
Each line MUST be:
MrDestructoid <ALL CAPS REWRITE> VoHiYo <the original URL exactly>

Hard limits:
- Each complete line <= 400 characters.
- Only letters, spaces, digits, and ordinary punctuation.
- No quotes, no numbering, no extra commentary.
"""


def read_existing(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def main():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY missing")

    articles = load_articles()
    print(f"articles: {len(articles)}")
    if len(articles) < 8:
        raise SystemExit("not enough articles")

    # Prefer 48 distinct items; wrap if feeds are thin
    picked = []
    i = 0
    while len(picked) < 48:
        picked.append(articles[i % len(articles)])
        i += 1

    lines = []
    batch_size = 6
    for start in range(0, 48, batch_size):
        batch = picked[start:start + batch_size]
        try:
            raw = gemini(api_key, batch_prompt(batch))
        except Exception as exc:
            print(f"batch {start} failed: {exc}")
            continue
        raw_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        raw_lines = [ln for ln in raw_lines if "MrDestructoid" in ln or "VOHIYO" in ln.upper()]
        for art, raw_line in zip(batch, raw_lines):
            fixed = normalize_line(raw_line, art["link"])
            if fixed:
                lines.append(fixed)
        print(f"batch {start}: {len(lines)} total")
        time.sleep(1)

    existing = read_existing("stories.txt")
    seen = set(lines)
    for old in existing:
        if len(lines) >= 48:
            break
        if old not in seen:
            lines.append(old)
            seen.add(old)

    if len(lines) < 48:
        raise SystemExit(f"only produced {len(lines)} stories")

    lines = lines[:48]
    with open("stories.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote stories.txt")


if __name__ == "__main__":
    main()
