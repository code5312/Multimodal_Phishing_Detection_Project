#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust URL collector for phishing dataset (Docker/Linux friendly)

- 입력 URL 리스트를 빠르게 프리체크(HEAD/GET 3s) 후 Playwright로 렌더링
- HTML 저장, 메타 JSON/CSV 기록, (옵션) 스크린샷 저장
- eTLD+1 / HTTPS / 상태코드 / 텍스트 길이 등 요약 필드
- User-Agent 로테이션, 컨텍스트 주기적 리셋, 타임아웃/재시도 가드
- --label 로 정상(0)/피싱(1) 라벨을 메타/CSV에 동시 기록

Usage:
  python -u collector/collect_phish.py --input urls/phish.txt --out data/collected \
    --label 1 --limit 5000 --retries 0 --timeout 10000 --delay 0.3 --headless --screenshot
"""
import os, csv, json, time, argparse, re, sys
from urllib.parse import urlparse
from typing import Tuple

import requests
import tldextract
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

# ----------------------------- utils -----------------------------

HTTP_RE = re.compile(r"^https?://", re.I)

def safe_filename(s: str, max_len: int = 140) -> str:
    s = re.sub(r'[^A-Za-z0-9\-_.]', '_', s)
    return s[:max_len]

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_csv(path: str, row: dict, header: list) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)

def etld1(url: str) -> str:
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else (ext.domain or "")

def quick_precheck(url: str, timeout: float = 3.0) -> Tuple[bool, str]:
    """빠른 HEAD/GET 프리체크 (3s). 비 HTML/죽은 URL은 바로 컷."""
    if not HTTP_RE.match(url or ""):
        return False, "not_http"
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code >= 400:
            return False, f"head_{r.status_code}"
        ct = (r.headers.get("content-type") or "").lower()
        if ("text/html" not in ct) and ("application/xhtml" not in ct) and ("text/plain" not in ct):
            # HTML이 아니어도 일부 피싱은 text/plain로 제공 → 허용 범위 약간 확장
            return False, f"ctype_{ct[:40]}"
        return True, "ok"
    except Exception:
        try:
            r = requests.get(url, timeout=timeout, allow_redirects=True, stream=True)
            if r.status_code >= 400:
                return False, f"get_{r.status_code}"
            return True, "ok_get"
        except Exception as e:
            return False, f"precheck_{e.__class__.__name__}"

# ----------------------------- fetch -----------------------------

UA_POOL = [
    # 간단한 로테이션용 UA (필요 시 아무거나 더 추가)
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36 Edg/120.0",
]

def extract_text_len(html: str) -> int:
    # 텍스트 길이만 빠르게 필요 → 정규식 기반
    txt = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.I|re.S)
    txt = re.sub(r"<style.*?>.*?</style>", " ", txt, flags=re.I|re.S)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return len(txt)

def fetch_one(page, url: str, outdir: str, idx: int, label: int, timeout: int,
              take_screenshot: bool = False):
    rec = {
        "url": url,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index": idx,
        "label": label,
    }
    try:
        response = page.goto(url, timeout=timeout, wait_until="domcontentloaded")
        status = None
        if response:
            try:
                status = response.status
            except Exception:
                status = None
        rec["status"] = status

        # 네트워크 idle 최대 5s까지 대기 (best-effort)
        try:
            page.wait_for_load_state("networkidle", timeout=min(5000, timeout))
        except PWTimeoutError:
            pass

        html = page.content()
        ident = f"{idx}_{safe_filename(url)}"
        sample_dir = os.path.join(outdir, ident)
        ensure_dir(sample_dir)
        html_path = os.path.join(sample_dir, "page.html")
        with open(html_path, "w", encoding="utf-8") as hf:
            hf.write(html)

        if take_screenshot:
            try:
                shot_path = os.path.join(sample_dir, "shot.png")
                page.screenshot(path=shot_path, full_page=True)
                rec["screenshot"] = shot_path
            except Exception:
                pass

        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else (ext.domain or "")

        rec.update({
            "html_path": html_path,
            "text_len": extract_text_len(html),
            "domain": domain,
            "is_https": urlparse(url).scheme.lower() == "https",
        })

        meta_path = os.path.join(sample_dir, "page.meta.json")
        write_json(meta_path, rec)
        return rec
    except PWTimeoutError as e:
        rec["error"] = f"TimeoutError: {e}"
        return rec
    except Exception as e:
        rec["error"] = f"{e.__class__.__name__}: {e}"
        return rec

# ----------------------------- main -----------------------------

def main(input_file: str, outdir: str, limit: int, retries: int, timeout: int,
         headless: bool, delay: float, user_agent: str, screenshot: bool, label: int,
         reset_every: int):
    # URL 로드
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        urls = [l.strip() for l in f if l.strip()]
    if limit and limit > 0:
        urls = urls[:limit]
    ensure_dir(outdir)

    # CSV 준비
    csv_path = os.path.join(outdir, "url_data.csv")
    csv_header = [
        "url", "label", "html_path", "domain", "is_https", "status",
        "text_len", "fetched_at", "index", "error", "attempts", "precheck"
    ]

    print(f"Starting fetch: {len(urls)} urls, outdir={outdir}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        # 초기 컨텍스트
        ua0 = user_agent or UA_POOL[0]
        context = browser.new_context(user_agent=ua0)
        page = context.new_page()

        for idx, url in enumerate(urls, start=1):
            # 절대 URL만 처리
            if not HTTP_RE.match(url):
                print(f"[SKIP] Not an absolute http(s) URL: {url}")
                append_csv(csv_path, {
                    "url": url, "label": label, "html_path": "", "domain": "",
                    "is_https": False, "status": "", "text_len": 0,
                    "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "index": idx, "error": "not_http", "attempts": 0, "precheck": "fail"
                }, header=csv_header)
                continue

            # N개마다 UA 로테이션/컨텍스트 리셋
            if reset_every > 0 and (idx % reset_every == 0):
                try:
                    page.close()
                except Exception:
                    pass
                try:
                    context.close()
                except Exception:
                    pass
                ua = user_agent or UA_POOL[(idx // reset_every) % len(UA_POOL)]
                context = browser.new_context(user_agent=ua)
                page = context.new_page()

            # 프리체크 3초
            ok, why = quick_precheck(url, timeout=3.0)
            if not ok:
                print(f"[{idx}/{len(urls)}] Precheck FAIL {url} -> {why}")
                append_csv(csv_path, {
                    "url": url, "label": label, "html_path": "", "domain": etld1(url),
                    "is_https": urlparse(url).scheme.lower() == "https",
                    "status": "", "text_len": 0, "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "index": idx, "error": f"precheck:{why}", "attempts": 0, "precheck": "fail"
                }, header=csv_header)
                time.sleep(min(delay, 1.0))
                continue

            attempts = 0
            rec = None
            while attempts <= max(0, retries):
                attempts += 1
                print(f"[{idx}/{len(urls)}] Fetching {url} (try {attempts})")
                r = fetch_one(page, url, outdir, idx, label, timeout, take_screenshot=screenshot)
                r["attempts"] = attempts
                r["precheck"] = "ok"
                rec = r
                if "html_path" in r:
                    print(" -> OK")
                    break
                else:
                    print(" -> ERROR", r.get("error", ""))
                    if attempts <= retries:
                        time.sleep(0.6)
                        continue
                    else:
                        break

            # CSV 기록
            append_csv(csv_path, {
                "url": rec.get("url"),
                "label": label,
                "html_path": rec.get("html_path", ""),
                "domain": rec.get("domain", ""),
                "is_https": rec.get("is_https", False),
                "status": rec.get("status", ""),
                "text_len": rec.get("text_len", 0),
                "fetched_at": rec.get("fetched_at", ""),
                "index": rec.get("index", idx),
                "error": rec.get("error", ""),
                "attempts": rec.get("attempts", attempts),
                "precheck": rec.get("precheck", ""),
            }, header=csv_header)

            # polite delay
            if delay and delay > 0:
                time.sleep(delay)

        # clean up
        try:
            page.close()
        except Exception:
            pass
        try:
            context.close()
        except Exception:
            pass
        browser.close()

    # 요약
    summary_path = os.path.join(outdir, "fetch_summary.json")
    # 큰 리스트 저장 대신 간단 지표만 남김(파일 커지는 것 방지)
    stats = {"total_urls": len(urls), "outdir": outdir, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(summary_path, stats)
    print("Done. Summary JSON:", summary_path)
    print("CSV summary:", csv_path)

# ----------------------------- CLI -----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to URL list (one per line)")
    ap.add_argument("--out", required=True, help="Output dir to store collected samples")
    ap.add_argument("--label", type=int, default=-1, help="Label to stamp into meta/CSV (0=benign,1=phish)")
    ap.add_argument("--limit", type=int, default=10000, help="Max URLs to fetch")
    ap.add_argument("--retries", type=int, default=0, help="Retries per URL on failure")
    ap.add_argument("--timeout", type=int, default=10000, help="Page goto timeout in ms")
    ap.add_argument("--headless", action="store_true", help="Run headless")
    ap.add_argument("--user-agent", default="", help="Override UA (otherwise rotate preset)")
    ap.add_argument("--delay", type=float, default=0.3, help="Delay between URLs (seconds)")
    ap.add_argument("--screenshot", action="store_true", help="Save screenshots as shot.png")
    ap.add_argument("--reset-every", type=int, default=300, help="Reset context/page every N urls (0=never)")
    args = ap.parse_args()

    main(
        input_file=args.input,
        outdir=args.out,
        limit=args.limit,
        retries=args.retries,
        timeout=args.timeout,
        headless=args.headless,
        delay=args.delay,
        user_agent=args.user_agent,
        screenshot=args.screenshot,
        label=args.label,
        reset_every=args.reset_every,
    )
