# tools/make_url_lists.py
# -*- coding: utf-8 -*-
"""
피싱/정상 URL 리스트 생성 스크립트 (통합 최종본)
- 피싱: OpenPhish + (옵션) URLhaus + (옵션) PhishTank CSV + (옵션) extra 파일들
       → 필터(IP/비표준포트/바이너리/악성 경로) → eTLD+1 중복 제거 → 셔플 → limit → 저장
- 정상: Tranco CSV → https://<domain> → eTLD+1 중복 제거 → 셔플 → limit → 저장

추가 기능:
- --fetch: 로컬에 openphish/urlhaus 원본 텍스트가 없으면 원격에서 받아 저장
- --no-urlhaus: URLhaus 배제
- 필터 토글: --allow-ip / --allow-nonstd-port / --allow-binary / --allow-badpath
- PhishTank CSV 병합: --phishtank-csv
- 추가 파일 병합: --extra <파일들>

요구 패키지: requests, tldextract, pandas
"""

import os
import re
import csv
import time
import argparse
import random
import tldextract
import pandas as pd
from typing import List

try:
    import requests
except Exception:
    requests = None

# ---------- 소스 URL ----------
OPENPHISH_TXT = "https://openphish.com/feed.txt"
URLHAUS_TXT   = "https://urlhaus.abuse.ch/downloads/text/"

# ---------- 정규식 ----------
HTTP_RE   = re.compile(r"^https?://", re.I)
IP_RE     = re.compile(r"^https?://\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?/", re.I)
# 80/443 외 포트 붙으면 제외
BAD_PORT_RE = re.compile(r"^https?://[^/]+:(?!80(?:/|$)|443(?:/|$))\d{2,5}(?:/|$)", re.I)
# 바이너리/스크립트 확장자
BAD_EXT_RE  = re.compile(r"\.(exe|scr|dll|bin|sh|bat|ps1|apk|msi|com|dat)(?:\?|$)", re.I)
# 악성 경로 패턴
BAD_PATH_RE = re.compile(r"/(?:sshd|tftp|bin\.sh)(?:\?|$|/)", re.I)

# ---------- 유틸 ----------
def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def clean_line(line: str) -> str:
    line = line.strip()
    if not line or line.startswith("#"):
        return ""
    if "#" in line:
        line = line.split("#", 1)[0].strip()
    return line

def read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = clean_line(raw)
            if line:
                out.append(line)
    return out

def fetch_text(url: str, timeout: int = 25) -> str:
    if requests is None:
        raise RuntimeError("requests 미설치: pip install requests")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

def normalize_urls(lines: List[str]) -> List[str]:
    return [l for l in lines if HTTP_RE.match(l)]

def dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen: 
            continue
        seen.add(x); out.append(x)
    return out

def etld1(url: str) -> str:
    e = tldextract.extract(url)
    return (e.domain + "." + e.suffix) if e.suffix else e.domain or ""

def dedup_by_domain(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        d = etld1(u)
        if not d or d in seen:
            continue
        seen.add(d); out.append(u)
    return out

def maybe_shuffle(urls: List[str], seed: int = 42) -> List[str]:
    rnd = random.Random(seed)
    rnd.shuffle(urls)
    return urls

def save_list(path: str, urls: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(u.strip() for u in urls))

# ---------- 소스 로더 ----------
def from_openphish_local_or_fetch(fetch_missing: bool) -> List[str]:
    loc = os.path.join("urls", "phish_openphish.txt")
    if not os.path.exists(loc) and fetch_missing:
        try:
            text = fetch_text(OPENPHISH_TXT)
            ensure_dir(os.path.dirname(loc))
            with open(loc, "w", encoding="utf-8") as f:
                f.write(text)
            print("[OK] fetched openphish ->", loc)
        except Exception as e:
            print("[WARN] openphish fetch failed:", e)
    return normalize_urls(read_lines(loc))

def from_urlhaus_local_or_fetch(fetch_missing: bool) -> List[str]:
    loc = os.path.join("urls", "malicious_urlhaus.txt")
    if not os.path.exists(loc) and fetch_missing:
        try:
            text = fetch_text(URLHAUS_TXT)
            ensure_dir(os.path.dirname(loc))
            with open(loc, "w", encoding="utf-8") as f:
                f.write(text)
            print("[OK] fetched urlhaus ->", loc)
        except Exception as e:
            print("[WARN] urlhaus fetch failed:", e)
    return normalize_urls(read_lines(loc))

def from_phishtank_csv(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    urls = []
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            for cell in row:
                cell = (cell or "").strip()
                if HTTP_RE.match(cell):
                    urls.append(cell)
    return urls

def from_custom_file(path: str) -> List[str]:
    return normalize_urls(read_lines(path))

# ---------- 피싱용 필터 ----------
def is_web_phish_like(u: str,
                      allow_ip: bool = False,
                      allow_nonstd_port: bool = False,
                      allow_binary: bool = False,
                      allow_badpath: bool = False) -> bool:
    if not allow_ip and IP_RE.search(u): return False
    if not allow_nonstd_port and BAD_PORT_RE.search(u): return False
    if not allow_binary and BAD_EXT_RE.search(u): return False
    if not allow_badpath and BAD_PATH_RE.search(u): return False
    return True

# ---------- 빌더 ----------
def build_phish(out_path: str,
                limit: int = 2000,
                seed: int = 42,
                use_openphish: bool = True,
                use_urlhaus: bool = True,
                phishtank_csv: str = None,
                extra_files: List[str] = None,
                fetch_missing: bool = False,
                diversify: bool = True,
                allow_ip: bool = False,
                allow_nonstd_port: bool = False,
                allow_binary: bool = False,
                allow_badpath: bool = False) -> None:

    all_urls: List[str] = []

    if use_openphish:
        print("[INFO] using OpenPhish (local or fetched)")
        all_urls += from_openphish_local_or_fetch(fetch_missing)

    if use_urlhaus:
        print("[INFO] using URLhaus (local or fetched)")
        all_urls += from_urlhaus_local_or_fetch(fetch_missing)

    if phishtank_csv:
        try:
            print("[INFO] reading PhishTank CSV:", phishtank_csv)
            all_urls += from_phishtank_csv(phishtank_csv)
        except Exception as e:
            print("[WARN] PhishTank parse failed:", e)

    if extra_files:
        for p in extra_files:
            try:
                print("[INFO] loading extra:", p)
                all_urls += from_custom_file(p)
            except Exception as e:
                print("[WARN] extra file skipped:", p, e)

    # 1) 절대 URL만
    all_urls = [u for u in all_urls if HTTP_RE.match(u)]
    # 2) 필터
    before = len(all_urls)
    all_urls = [u for u in all_urls if is_web_phish_like(
        u,
        allow_ip=allow_ip,
        allow_nonstd_port=allow_nonstd_port,
        allow_binary=allow_binary,
        allow_badpath=allow_badpath
    )]
    print(f"[INFO] filter: {before} -> {len(all_urls)}")

    # 3) 중복 제거
    all_urls = dedup_preserve_order(all_urls)
    # 4) eTLD+1 다양화
    if diversify:
        all_urls = dedup_by_domain(all_urls)
    # 5) 셔플 + 개수 제한
    all_urls = maybe_shuffle(all_urls, seed=seed)
    if limit:
        all_urls = all_urls[:int(limit)]

    save_list(out_path, all_urls)
    print(f"[OK] wrote {len(all_urls)} urls -> {out_path}")

def build_benign_from_tranco(tranco_csv: str,
                             out_path: str,
                             limit: int = 2000,
                             seed: int = 42,
                             diversify: bool = True,
                             scheme: str = "https") -> None:
    if not os.path.exists(tranco_csv):
        raise FileNotFoundError(tranco_csv)

    df = pd.read_csv(tranco_csv)
    col = None
    for c in ["domain", "site"]:
        if c in df.columns:
            col = c; break
    if not col:
        col = df.columns[0]

    domains = [str(x).strip() for x in df[col].astype(str).tolist() if str(x).strip()]
    urls = [f"{scheme}://{d}" for d in domains if d]

    urls = dedup_preserve_order(urls)
    if diversify:
        urls = dedup_by_domain(urls)
    urls = maybe_shuffle(urls, seed=seed)
    if limit:
        urls = urls[:int(limit)]

    save_list(out_path, urls)
    print(f"[OK] wrote {len(urls)} urls -> {out_path}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Build URL lists for phishing/benign with filters & extras")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # phish
    sp_phish = sub.add_parser("phish", help="Build phishing URL list")
    sp_phish.add_argument("--out", default="urls/phish.txt")
    sp_phish.add_argument("--limit", type=int, default=2000)
    sp_phish.add_argument("--seed", type=int, default=42)
    sp_phish.add_argument("--no-openphish", action="store_true", help="OpenPhish 제외")
    sp_phish.add_argument("--no-urlhaus", action="store_true", help="URLhaus 제외")
    sp_phish.add_argument("--fetch", action="store_true", help="로컬 원본 없으면 원격에서 다운로드")
    sp_phish.add_argument("--phishtank-csv", help="로컬 PhishTank CSV 경로 (선택)")
    sp_phish.add_argument("--extra", nargs="*", help="추가 텍스트/CSV 파일들 (선택)")
    sp_phish.add_argument("--no-diversify", action="store_true", help="eTLD+1 중복 제거 비활성화")
    # 필터 토글
    sp_phish.add_argument("--allow-ip", action="store_true", help="IP URL 허용")
    sp_phish.add_argument("--allow-nonstd-port", action="store_true", help="비표준 포트 허용")
    sp_phish.add_argument("--allow-binary", action="store_true", help="바이너리 확장자 허용")
    sp_phish.add_argument("--allow-badpath", action="store_true", help="악성 경로 패턴 허용")

    # benign
    sp_ben = sub.add_parser("benign", help="Build benign URL list from Tranco CSV")
    sp_ben.add_argument("--tranco-csv", required=True)
    sp_ben.add_argument("--out", default="urls/benign.txt")
    sp_ben.add_argument("--limit", type=int, default=2000)
    sp_ben.add_argument("--seed", type=int, default=42)
    sp_ben.add_argument("--no-diversify", action="store_true", help="eTLD+1 중복 제거 비활성화")
    sp_ben.add_argument("--scheme", default="https", choices=["http", "https"])

    args = ap.parse_args()

    if args.cmd == "phish":
        build_phish(
            out_path=args.out,
            limit=args.limit,
            seed=args.seed,
            use_openphish=not args.no_openphish,
            use_urlhaus=not args.no_urlhaus,
            phishtank_csv=args.phishtank_csv,
            extra_files=args.extra,
            fetch_missing=args.fetch,
            diversify=not args.no_diversify,
            allow_ip=args.allow_ip,
            allow_nonstd_port=args.allow_nonstd_port,
            allow_binary=args.allow_binary,
            allow_badpath=args.allow_badpath
        )
    elif args.cmd == "benign":
        build_benign_from_tranco(
            tranco_csv=args.tranco_csv,
            out_path=args.out,
            limit=args.limit,
            seed=args.seed,
            diversify=not args.no_diversify,
            scheme=args.scheme
        )

if __name__ == "__main__":
    main()
