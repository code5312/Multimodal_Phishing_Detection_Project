"""
피싱 사이트 특성 추출기
수집된 데이터에서 피싱 탐지를 위한 특성들을 추출합니다.
"""

"""
extract_features.py
- Walks collected data folder
- Parses page.meta.json and page.html
- Extracts URL-based numeric features, HTML structural features, and a TF-IDF vector for text
- Performs optional TruncatedSVD to reduce TF-IDF dims
- Writes fusion_input.csv with columns: id, url, label (if present), <url features>, <html features>, <text_svd_0..N>
Usage:
  python extract_features.py --collected ../data/collected --out ../data/fusion_input.csv --svd_components 200
"""
import os, json, argparse, re, math
from collections import Counter
from urllib.parse import urlparse

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# -----------------------
# Feature helpers
# -----------------------
def host_entropy(s: str) -> float:
    if not s:
        return 0.0
    cnt = Counter(s)
    probs = [v / len(s) for v in cnt.values()]
    return float(-sum(p * math.log2(p) for p in probs)) if probs else 0.0

def extract_url_features(url: str) -> dict:
    u = urlparse(url)
    hostname = u.hostname or ""
    path = u.path or ""
    features = {
        'url_len': len(url),
        'hostname_len': len(hostname),
        'path_len': len(path),
        'num_digits': sum(c.isdigit() for c in url),
        'num_hyphen': url.count('-'),
        'num_underscore': url.count('_'),
        'num_dots': hostname.count('.'),
        'has_at': int('@' in url),
        'is_ip': 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname) else 0,
        'is_https': 1 if (u.scheme == 'https') else 0,
        'host_entropy': host_entropy(hostname),
        'path_depth': len([p for p in path.split('/') if p]),
    }
    suspicious = ['login','secure','account','update','verify','bank','confirm','signin','password']
    lo = url.lower()
    features['has_suspicious_kw'] = int(any(k in lo for k in suspicious))
    features['suspicious_token_count'] = sum(1 for t in suspicious if t in lo)
    return features

def extract_html_struct_features(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=' ', strip=True)

    feats = {
        'text_len': len(text),
        'num_forms': len(soup.find_all('form')),
        'num_inputs': len(soup.find_all('input')),
        'num_links': len(soup.find_all('a')),
        'num_scripts': len(soup.find_all('script')),
        'num_iframes': len(soup.find_all('iframe')),
        'has_hidden_styles': int(bool(re.search(r'display\s*:\s*none|visibility\s*:\s*hidden', html, re.I))),
        'has_login_keywords': int(bool(re.search(r'login|signin|password|아이디|비밀번호|계정', text, re.I))),
    }
    scripts_src = [s.get('src') for s in soup.find_all('script') if s.get('src')]
    feats['external_script_count'] = sum(1 for s in scripts_src if str(s).startswith(('http://','https://')))
    return feats

# -----------------------
# Collection loader
# -----------------------
def collect_samples(collected_dir: str):
    samples = []
    for d in sorted(os.listdir(collected_dir)):  # reproducible ordering
        sd = os.path.join(collected_dir, d)
        if not os.path.isdir(sd):
            continue
        meta = os.path.join(sd, "page.meta.json")
        htmlf = os.path.join(sd, "page.html")
        if not (os.path.exists(meta) and os.path.exists(htmlf)):
            continue

        with open(meta, "r", encoding="utf-8") as mf:
            meta_j = json.load(mf)
        with open(htmlf, "r", encoding="utf-8", errors="ignore") as hf:
            html = hf.read()

        # prefer final_url (redirected)
        url = meta_j.get("final_url") or meta_j.get("url", "")

        # robust text extraction
        try:
            from trafilatura import extract
            text = extract(html) or ""
        except Exception:
            text = re.sub("<[^<]+?>", " ", html)

        samples.append({
            "id": d,
            "url": url,
            "domain": meta_j.get("domain", ""),
            "is_https": meta_j.get("is_https", False),
            "status": meta_j.get("status", None),
            "text": text,
            "html": html
        })
    return samples

# -----------------------
# Main
# -----------------------
def main(collected_dir, out_csv, svd_components=200, max_tfidf_features=5000):
    samples = collect_samples(collected_dir)
    print(f"[INFO] Collected {len(samples)} valid samples")
    if len(samples) == 0:
        print("No samples found. Exiting.")
        return

    # build lists
    ids, urls, texts = [], [], []
    url_feats, html_feats = [], []
    for s in samples:
        ids.append(s['id'])
        urls.append(s['url'])
        texts.append((s['text'] or "")[:100000])  # truncate super long
        url_feats.append(extract_url_features(s['url'] or ""))
        html_feats.append(extract_html_struct_features(s['html'] or ""))

    df_url  = pd.DataFrame(url_feats)
    df_html = pd.DataFrame(html_feats)

    # TF-IDF on raw text (for numeric baselines / SVD features)
    texts = [t if (t and t.strip()) else "" for t in texts]
    if all(len(t.strip()) == 0 for t in texts):
        print("[WARN] All texts are empty. Injecting a dummy token to avoid empty vocabulary.")
        texts = ["dummy"] * len(texts)

    vectorizer = TfidfVectorizer(max_features=max_tfidf_features, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(texts)
    print("[INFO] TF-IDF shape:", X_tfidf.shape)
    n_feats = X_tfidf.shape[1]

    # SVD guard (avoid dense blow-up)
    use_svd = bool(svd_components and svd_components > 0 and n_feats > 1)
    if not use_svd and n_feats > 1000:
        print("[WARN] High-dimensional TF-IDF without SVD; forcing SVD to avoid dense memory blow-up.")
        use_svd = True

    if use_svd:
        svd_n = min(svd_components if svd_components and svd_components > 0 else 200, n_feats - 1)
        svd = TruncatedSVD(n_components=svd_n, random_state=42)
        X_red = svd.fit_transform(X_tfidf)
        print("[INFO] SVD result shape:", X_red.shape)
        df_text = pd.DataFrame(X_red, columns=[f"svd_{i}" for i in range(X_red.shape[1])])
    else:
        df_text = pd.DataFrame(X_tfidf.toarray(), columns=[f"tfidf_{i}" for i in range(n_feats)])

    # meta cols (useful features too)
    meta_df = pd.DataFrame({
        "domain":   [s.get("domain","") for s in samples],
        "is_https": [int(bool(s.get("is_https", False))) for s in samples],
        "status":   [s.get("status", None) for s in samples],
        "text":     texts,  # keep raw text for multimodal training
    })

    # concat
    df_final = pd.concat(
        [
            pd.DataFrame({"id": ids, "url": urls}).reset_index(drop=True),
            meta_df.reset_index(drop=True),
            df_url.reset_index(drop=True),
            df_html.reset_index(drop=True),
            df_text.reset_index(drop=True),
        ],
        axis=1,
    )

    df_final.to_csv(out_csv, index=False)
    print("[INFO] Fusion CSV saved to", out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected", required=True, help="Collected dir (data/collected)")
    parser.add_argument("--out", required=True, help="Output fusion csv path")
    parser.add_argument("--svd_components", type=int, default=200)
    parser.add_argument("--max_tfidf_features", type=int, default=5000)
    args = parser.parse_args()
    main(args.collected, args.out, args.svd_components, args.max_tfidf_features)
