# -*- coding: utf-8 -*-
import argparse, csv, re, html
from urllib.parse import urlsplit, urlunsplit

HTTP_RE = re.compile(r"^https?://", re.I)

def normalize_url(u, prefer_https=True):
    if not u:
        return None
    u = html.unescape(u.strip())  # &amp; -> &
    if not u:
        return None
    # 스킴 없으면 붙이기
    if not HTTP_RE.match(u):
        scheme = "https" if prefer_https else "http"
        u = f"{scheme}://{u}"
    # 공백 제거
    u = u.replace(" ", "")
    # urlsplit/unsplit로 간단 정리
    sp = urlsplit(u)
    if not sp.netloc:
        return None
    return urlunsplit(sp)

# ... 위쪽 생략 ...
def main(csv_path, out_path, url_col=None, prefer_https=True, dedup=True):
    seen = set()
    kept = 0
    with open(csv_path, encoding="utf-8", errors="ignore", newline="") as f, \
         open(out_path, "w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        use_dict = True
        if reader.fieldnames is None:
            f.seek(0)
            rdr = csv.reader(f)
            use_dict = False

        if use_dict:
            # ✅ URL 컬럼 대소문자 무시 매핑
            field_map = {c.lower(): c for c in reader.fieldnames}
            if url_col:
                uc = field_map.get(url_col.lower(), url_col)  # 사용자가 준 이름을 실제 키로 매핑
            else:
                uc = field_map.get("url", reader.fieldnames[0])  # 'url' 없으면 첫 컬럼 사용

            for row in reader:
                u = row.get(uc, "")
                nu = normalize_url(u, prefer_https=prefer_https)
                if not nu:
                    continue
                if dedup and nu in seen:
                    continue
                seen.add(nu)
                out.write(nu + "\n")
                kept += 1
        else:
            for row in rdr:
                if not row:
                    continue
                u = row[0]
                nu = normalize_url(u, prefer_https=prefer_https)
                if not nu:
                    continue
                if dedup and nu in seen:
                    continue
                seen.add(nu)
                out.write(nu + "\n")
                kept += 1
    print(f"[OK] wrote {kept} urls -> {out_path}")

