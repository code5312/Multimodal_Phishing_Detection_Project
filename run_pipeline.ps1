Param(
  [string]$PythonExe        = "python",
  [string]$VenvPath         = ".\venv",
  [string]$OutDir           = "models\out",
  [string]$CollectedDir     = "data\collected",
  [string]$UrlsDir          = "urls",
  [int]$PhishLimit          = 2000,
  [int]$BenignLimit         = 2000,
  [string]$TrancoCsv        = "urls\tranco.csv",
  [switch]$SkipUrlBuild,
  [switch]$Headless,
  [int]$TimeoutMs           = 20000,
  [float]$DelaySec          = 1.0
)

Set-Location -Path $PSScriptRoot
$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n==== $msg ====" -ForegroundColor Cyan }
function Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Gray }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Die($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# 0) 폴더 준비
Step "준비"
New-Item -ItemType Directory -Force -Path $UrlsDir, "data", $CollectedDir, "tools" | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $OutDir) | Out-Null

# 1) venv/의존성
Step "가상환경 & 의존성 설치"
if (!(Test-Path "$VenvPath\Scripts\python.exe")) { & $PythonExe -m venv $VenvPath }
& "$VenvPath\Scripts\Activate.ps1"
python -c "import sys; print('PYTHON:', sys.executable)"
pip install -U pip
if (Test-Path ".\requirements.txt") {
  pip install -r requirements.txt
} else {
  pip install playwright tldextract trafilatura beautifulsoup4 scikit-learn lightgbm shap pandas numpy matplotlib seaborn joblib requests regex
}
playwright install chromium

# 2) 최신 URL 리스트 생성
if (-not $SkipUrlBuild) {
  Step "최신 URL 리스트 생성"

  # 소스 텍스트(없을 때만 미리 받아두기 — 통합 스크립트의 --fetch를 쓸 거라 필수는 아님)
  try {
    Invoke-WebRequest "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt" -OutFile "$UrlsDir\phish_openphish.txt" -ErrorAction SilentlyContinue
    Invoke-WebRequest "https://urlhaus.abuse.ch/downloads/text/" -OutFile "$UrlsDir\malicious_urlhaus.txt" -ErrorAction SilentlyContinue
  } catch {
    Warn "피드 다운로드 일부 실패: $($_.Exception.Message) (make_url_lists.py가 --fetch로 보완 가능)"
  }

  # tools/make_url_lists.py 없으면 최소 버전 생성(구버전; 필터 옵션 없음)
  if (-not (Test-Path ".\tools\make_url_lists.py")) {
@'
import os, re, argparse, random, requests, tldextract, pandas as pd
HTTP_RE = re.compile(r"^https?://", re.I)
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def dedup_preserve_order(items):
    s=set(); out=[]
    for x in items:
        if x in s: continue
        s.add(x); out.append(x)
    return out
def etld1(u):
    e=tldextract.extract(u); return (e.domain+"."+e.suffix) if e.suffix else e.domain
def dedup_by_domain(urls):
    s=set(); out=[]
    for u in urls:
        d=etld1(u)
        if d and d not in s:
            s.add(d); out.append(u)
    return out
def from_file(path):
    if not os.path.exists(path): return []
    out=[]
    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line=raw.strip()
            if not line or line.startswith("#"): continue
            if "#" in line: line=line.split("#",1)[0].strip()
            if HTTP_RE.match(line): out.append(line)
    return out
def build_phish(out_path, limit=2000, seed=42):
    urls=[]
    urls+=from_file("urls/phish_openphish.txt")
    urls+=from_file("urls/malicious_urlhaus.txt")
    urls=dedup_preserve_order(urls)
    urls=dedup_by_domain(urls)
    random.Random(seed).shuffle(urls)
    urls=urls[:limit]
    ensure_dir(os.path.dirname(out_path))
    open(out_path,"w",encoding="utf-8").write("\n".join(urls))
    print("[OK] phish:", len(urls), "->", out_path)
def build_benign_from_tranco(tranco_csv, out_path, limit=2000, seed=42):
    if not os.path.exists(tranco_csv): raise FileNotFoundError(tranco_csv)
    df=pd.read_csv(tranco_csv)
    col="domain" if "domain" in df.columns else ("site" if "site" in df.columns else df.columns[0])
    domains=df[col].astype(str).tolist()
    urls=["https://"+d.strip() for d in domains if isinstance(d,str) and d.strip()]
    urls=dedup_preserve_order(urls); urls=dedup_by_domain(urls)
    random.Random(seed).shuffle(urls); urls=urls[:limit]
    ensure_dir(os.path.dirname(out_path))
    open(out_path,"w",encoding="utf-8").write("\n".join(urls))
    print("[OK] benign:", len(urls), "->", out_path)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["phish","benign"])
    ap.add_argument("--out", default="urls/phish.txt")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--tranco-csv"); ap.add_argument("--seed", type=int, default=42)
    a=ap.parse_args()
    if a.cmd=="phish": build_phish(a.out, a.limit, a.seed)
    else:
        if not a.tranco_csv: raise SystemExit("--tranco-csv is required for benign")
        build_benign_from_tranco(a.tranco_csv, a.out, a.limit, a.seed)
'@ | Set-Content -Path ".\tools\make_url_lists.py" -Encoding UTF8
  }

  # (신규) Tranco CSV 헤더 자동 보정: "순번,도메인" 형태인데 헤더가 없으면 rank,domain 헤더 삽입
  if (Test-Path $TrancoCsv) {
    $first = Get-Content $TrancoCsv -TotalCount 1
    $looksLikeNoHeader = ($first -match '^\s*\d+\s*,\s*[^,]+$') -and ($first -notmatch 'rank\s*,\s*domain')
    if ($looksLikeNoHeader) {
      Write-Host "[INFO] Tranco CSV에 헤더가 없어 보입니다. 헤더(rank,domain) 추가 중..." -ForegroundColor Yellow
      Copy-Item $TrancoCsv "$($TrancoCsv).bak" -Force
      @('rank,domain') + (Get-Content $TrancoCsv) | Set-Content $TrancoCsv -Encoding UTF8
      Write-Host "[OK] 헤더 추가 완료. 백업: $($TrancoCsv).bak" -ForegroundColor Green
    }
  }

  # 우리가 가진 make_url_lists.py가 '필터/옵션(--no-urlhaus, --fetch 등)'을 지원하는지 감지
  $ml = Get-Content ".\tools\make_url_lists.py" -Raw
  $supportsFilters = $ml -match '--no-urlhaus' -and $ml -match '--fetch'

  # 피싱 URL 생성: (우선) OpenPhish만, 필요시 --fetch 로 로컬 원본 없을 때 원격에서 다운로드
  if ($supportsFilters) {
    python tools\make_url_lists.py phish --out "$UrlsDir\phish.txt" --limit $PhishLimit --no-urlhaus --fetch
  } else {
    # 구버전(필터 미지원)용 폴백
    python tools\make_url_lists.py phish --out "$UrlsDir\phish.txt" --limit $PhishLimit
  }

  # benign URL 생성
  if (Test-Path $TrancoCsv) {
    python tools\make_url_lists.py benign --tranco-csv $TrancoCsv --out "$UrlsDir\benign.txt" --limit $BenignLimit
  } else {
    Warn "Tranco CSV($TrancoCsv) 없음 → benign.txt 생성 스킵"
  }
} else {
  Step "URL 리스트 생성 건너뜀 (--SkipUrlBuild)"
}


# 3) 수집 (한 줄, 백틱 제거)
Step "수집 시작 (Playwright)"
$headlessFlag = $null; if ($Headless) { $headlessFlag = "--headless" }

if (Test-Path "$UrlsDir\benign.txt") {
  python collector\collect_phish.py --input "$UrlsDir\benign.txt" --out $CollectedDir `
  --label 0 --limit $BenignLimit --retries 1 --timeout $TimeoutMs $headlessFlag --delay $DelaySec
} else { Warn "benign.txt 없음 → 정상군 수집 스킵" }

if (Test-Path "$UrlsDir\phish.txt") {
  python collector\collect_phish.py --input "$UrlsDir\phish.txt" --out $CollectedDir `
  --label 1 --limit $PhishLimit --retries 1 --timeout $TimeoutMs $headlessFlag --delay $DelaySec
} else { Die "phish.txt 없음 → 피싱 수집 불가" }

# 4) labels.csv 생성 (stdin 파이프)
Step "labels.csv 생성"
$pyScript = @'
import os, json, pandas as pd
rows=[]; base=r"data\collected"
if not os.path.exists(base): raise SystemExit("data\\collected not found")
for d in os.listdir(base):
    p=os.path.join(base,d,"page.meta.json")
    if os.path.exists(p):
        try:
            m=json.load(open(p,encoding="utf-8"))
            if "label" in m: rows.append({"id":d,"label":int(m["label"])})
        except Exception: pass
pd.DataFrame(rows).drop_duplicates("id").to_csv(r"data\labels.csv",index=False)
print("labels:",len(rows))
'@
$pyScript | & python -

# 5) 피처 결합
Step "피처 결합 CSV 생성"
python features\extract_features.py --collected $CollectedDir --out data\fusion.csv --svd_components 200 --max_tfidf_features 5000

# 6) 학습
Step "학습 시작"
python models\train_fusion.py --input data\fusion.csv --label-file data\labels.csv --out-dir $OutDir --text-column text --tfidf-max-features 5000 --lr-tfidf-max-features 80000 --lr-C-grid 0.01,0.03,0.1,0.3,1,3,10

Step "완료"; Write-Host "산출물 폴더: $OutDir" -ForegroundColor Green
