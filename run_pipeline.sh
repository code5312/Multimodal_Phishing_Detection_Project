#!/usr/bin/env bash
# End-to-end: URL 리스트 생성 -> 크롤링 -> 피처 추출 -> 학습
# Ubuntu 로컬(venv) 기준. Docker에서도 그대로 실행 가능(아래 안내 참고).

set -euo pipefail

# --------- 설정(필요 시 수정) ---------
PY=${PY:-python3}
USE_VENV=${USE_VENV:-1}
VENV=${VENV:-.venv}

URLS_DIR=${URLS_DIR:-urls}
COLLECTED=${COLLECTED:-data/collected}
OUT_DIR=${OUT_DIR:-models/out}
LOG_DIR=${LOG_DIR:-logs}

# 수집량
PHISH_LIMIT=${PHISH_LIMIT:-5000}
BENIGN_LIMIT=${BENIGN_LIMIT:-5000}

# 타임아웃/지연
TIMEOUT_MS=${TIMEOUT_MS:-10000}      # Playwright 네비게이션 타임아웃(ms)
WALL_TIMEOUT_MS=${WALL_TIMEOUT_MS:-10000}  # 한 URL 총 처리 시간 상한(ms)
DELAY_SEC=${DELAY_SEC:-0.25}

# 피처 설정
SVD_COMPONENTS=${SVD_COMPONENTS:-200}
TFIDF_MAX=${TFIDF_MAX:-30000}

# LR/LightGBM 학습 설정
LR_TFIDF_MAX=${LR_TFIDF_MAX:-80000}
LR_C_GRID=${LR_C_GRID:-"0.01,0.03,0.1,0.3,1,3,10"}

# 파일 경로(있으면 사용)
TRANCO_CSV=${TRANCO_CSV:-"$URLS_DIR/tranco.csv"}       # benign용 (선택)
PHISHTANK_CSV=${PHISHTANK_CSV:-"$URLS_DIR/phishtank.csv"} # phish 추가 소스(선택)

# --------- 유틸 ---------
step() { echo -e "\n==== $* ===="; }
info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }

mkdir -p "$URLS_DIR" "$COLLECTED" "$OUT_DIR" "$LOG_DIR" data tools

# --------- 0) 시스템 의존성(Playwright 브라우저 deps) ---------
step "Playwright 브라우저 의존성 점검(필요시 sudo)"
if command -v sudo >/dev/null 2>&1; then
  $PY -m pip --version >/dev/null 2>&1 || true
  # playwright 패키지가 아직 설치 전일 수 있으니 실패해도 무시
  $PY -m playwright install-deps || true
else
  warn "sudo 없음 → 수동으로 'sudo playwright install-deps' 를 나중에 실행 필요할 수 있음"
fi

# --------- 1) venv & 의존성 ---------
step "venv & deps"
if [ "$USE_VENV" -eq 1 ]; then
  if [ ! -d "$VENV" ]; then
    $PY -m venv "$VENV"
  fi
  # shellcheck disable=SC1090
  . "$VENV/bin/activate"
  PY=python
fi
$PY -m pip install -U pip wheel setuptools
if [ -f requirements.txt ]; then
  $PY -m pip install -r requirements.txt
fi
# 브라우저 설치(최초 1회)
$PY -m playwright install chromium

# --------- 2) Tranco 헤더 가드(없으면 자동 추가) ---------
if [ -f "$TRANCO_CSV" ]; then
  step "Tranco header guard"
  FIRST=$(head -n1 "$TRANCO_CSV" || true)
  # 형태: "rank,domain"이 없고 "순번,도메인" 꼴인 경우 헤더 주입
  if echo "$FIRST" | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*,[[:space:]]*[^,]+$' && \
     ! echo "$FIRST" | grep -qi '^rank,domain'; then
    cp "$TRANCO_CSV" "${TRANCO_CSV%.csv}_raw.csv"
    { echo 'rank,domain'; cat "$TRANCO_CSV"; } > "$TRANCO_CSV.tmp"
    mv "$TRANCO_CSV.tmp" "$TRANCO_CSV"
    info "Tranco 헤더 추가 완료 → $TRANCO_CSV (백업: ${TRANCO_CSV%.csv}_raw.csv)"
  fi
else
  warn "Tranco CSV($TRANCO_CSV) 없음 → benign 자동 생성은 생략될 수 있음"
fi

# --------- 3) URL 리스트 생성 ---------
step "URL 리스트 생성 (phish / benign)"
# tools/make_url_lists.py 가 있어야 함 (이전 단계에서 이미 받아둔 파일)
if [ ! -f tools/make_url_lists.py ]; then
  echo "[ERROR] tools/make_url_lists.py 가 없습니다." >&2
  exit 1
fi

# Phish 리스트: OpenPhish+URLhaus(+PhishTank CSV 있으면 추가), eTLD+1 다양화 & 셔플
PH_ARGS=(phish --out "$URLS_DIR/phish.txt" --limit "$PHISH_LIMIT")
[ -f "$PHISHTANK_CSV" ] && PH_ARGS+=(--phishtank-csv "$PHISHTANK_CSV")
$PY tools/make_url_lists.py "${PH_ARGS[@]}"

# Benign 리스트: Tranco CSV 있으면 생성
if [ -f "$TRANCO_CSV" ]; then
  $PY tools/make_url_lists.py benign \
     --tranco-csv "$TRANCO_CSV" \
     --out "$URLS_DIR/benign.txt" \
     --limit "$BENIGN_LIMIT"
else
  warn "Tranco CSV 미존재 → benign.txt 생성 스킵"
fi

# --------- 4) 크롤링 (benign → phish) ---------
# collect_phish.py 는 --wall-timeout 지원(이미 반영돼 있어야 함)
step "크롤링 시작 (benign)"
if [ -f "$URLS_DIR/benign.txt" ]; then
  $PY collector/collect_phish.py \
     --input "$URLS_DIR/benign.txt" --out "$COLLECTED" \
     --label 0 --limit "$BENIGN_LIMIT" \
     --retries 0 --timeout "$TIMEOUT_MS" --wall-timeout "$WALL_TIMEOUT_MS" \
     --delay "$DELAY_SEC" --headless \
     | tee "$LOG_DIR/collect_benign.log"
else
  warn "benign.txt 없음 → benign 수집 스킵"
fi

step "크롤링 시작 (phish)"
if [ -f "$URLS_DIR/phish.txt" ]; then
  $PY collector/collect_phish.py \
     --input "$URLS_DIR/phish.txt" --out "$COLLECTED" \
     --label 1 --limit "$PHISH_LIMIT" \
     --retries 0 --timeout "$TIMEOUT_MS" --wall-timeout "$WALL_TIMEOUT_MS" \
     --delay "$DELAY_SEC" --headless \
     | tee "$LOG_DIR/collect_phish.log"
else
  echo "[ERROR] phish.txt 없음 → 수집 불가" >&2
  exit 1
fi

# --------- 5) labels.csv 생성 ---------
step "labels.csv 생성"
$PY - <<'PY'
import os, json, pandas as pd
rows=[]; base="data/collected"
if os.path.isdir(base):
    for d in os.listdir(base):
        p=os.path.join(base,d,"page.meta.json")
        if os.path.exists(p):
            try:
                m=json.load(open(p,encoding="utf-8"))
                if "label" in m: rows.append({"id":d,"label":int(m["label"])})
            except: pass
pd.DataFrame(rows).drop_duplicates("id").to_csv("data/labels.csv", index=False)
print("labels:", len(rows))
PY

# --------- 6) 피처 추출 (TF-IDF+SVD, URL/HTML 구조 피처) ---------
step "피처 추출"
$PY features/extract_features.py \
   --collected "$COLLECTED" --out data/fusion.csv \
   --svd_components "$SVD_COMPONENTS" --max_tfidf_features "$TFIDF_MAX" \
   | tee "$LOG_DIR/extract_features.log"

# --------- 7) 학습 (도메인 그룹 분할 70/15/15, LGBM + LR baseline) ---------
step "모델 학습"
$PY models/train_fusion.py \
   --input data/fusion.csv --label-file data/labels.csv \
   --out-dir "$OUT_DIR" \
   --text-column text --tfidf-max-features "$TFIDF_MAX" \
   --lr-tfidf-max-features "$LR_TFIDF_MAX" \
   --lr-C-grid "$LR_C_GRID" \
   | tee "$LOG_DIR/train.log"

step "완료"
echo "산출물 폴더: $OUT_DIR"
