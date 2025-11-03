<div align="center">

<img alt="shield" src="https://img.shields.io/badge/Project-Phishing%20Multi‑Modal-blue" />
<img alt="python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
<img alt="lightgbm" src="https://img.shields.io/badge/LightGBM-✓-28a745" />
<img alt="playwright" src="https://img.shields.io/badge/Playwright-✓-00b894" />

<h2>멀티모달 피싱 탐지 (URL + HTML)</h2>
<p>수집 → 피처화 → 학습 → 설명(SHAP)까지 한 번에</p>

</div>

---
venv 상태에서 실행하기
## 🔰 빠른 시작

```bash
# 1) 수집
python collector/collect_phish.py --input urls/urls_to_fetch.txt --out data/collected \
  --limit 200 --retries 1 --timeout 20000 --headless --delay 1.0

# 2) 피처 결합 CSV 생성
python features/extract_features.py --collected data/collected --out data/fusion.csv \
  --svd_components 200 --max_tfidf_features 5000

# 3) 모델 학습
python models/train_fusion.py --input data/fusion.csv --label-file data/labels.csv \
  --out-dir models/out

# 4) 설명(SHAP)
python explain/explain_shap.py --model models/out/lgb_all_numeric.joblib \
  --x-csv models/out/X_test.csv --out explain/out
```

> 참고: 최초 1회 `playwright install` 필요

---

## ⚙️ 설치

```bash
python -m venv .venv
# Win: .venv\Scripts\activate   /   macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
playwright install
```

<details>
<summary><strong>핵심 패키지 버전</strong></summary>

```
playwright==1.40.0   tldextract==5.1.0   trafilatura==1.6.0   beautifulsoup4==4.12.3
scikit-learn==1.4.2  lightgbm==4.3.0     shap==0.42.1         numpy==1.26.4
scipy==1.12.0        pandas==2.2.2       joblib==1.4.2
matplotlib==3.8.4    seaborn==0.13.2     tqdm==4.66.4         requests==2.32.3
```

</details>

---

## 🗂 프로젝트 구조

```text
.
├─ collector/collect_phish.py        # URL 렌더링·HTML 저장·메타·CSV 요약
├─ features/extract_features.py      # URL/HTML 피처 + 텍스트 TF-IDF/SVD → fusion.csv
├─ models/train_fusion.py            # URL-only / HTML-only / All-numeric 학습
├─ explain/explain_shap.py           # 모델 번들 기반 SHAP 설명
├─ data/
│  ├─ collected/                     # 수집 산출물(id/page.html, page.meta.json)
│  ├─ fusion.csv                     # 피처 결합 CSV(학습 입력)
│  └─ labels.csv                     # id,label (0 정상 / 1 피싱)
├─ urls/urls_to_fetch.txt            # 한 줄당 1 URL
└─ requirements.txt  README.md
```

---

## 🧭 사용법

### 1) 수집(Collector)
- 렌더링된 HTML, 상태코드, 도메인, HTTPS 여부, 텍스트 길이 등 메타 저장

```bash
python collector/collect_phish.py --input urls/urls_to_fetch.txt --out data/collected \
  --limit 200 --retries 1 --timeout 20000 --headless --delay 1.0
```

### 2) 피처 생성(Features)
- URL: 길이/숫자/하이픈/점/@/IP/HTTPS/엔트로피/깊이/의심 키워드 등
- HTML: form/input/a/script/iframe 수, 숨김 스타일, 로그인 키워드, 외부 스크립트 수 등
- 텍스트: TF-IDF → SVD(고차원 방지)

```bash
python features/extract_features.py --collected data/collected --out data/fusion.csv \
  --svd_components 200 --max_tfidf_features 5000
```

### 3) 학습(Models)
- URL-only, HTML-only, All-numeric 3종 모델 학습 및 산출물 저장

```bash
python models/train_fusion.py --input data/fusion.csv --label-file data/labels.csv \
  --out-dir models/out
```

### 4) 설명(SHAP)
- 학습 번들(`model`, `feature_names`) 로드 → SHAP 요약/중요도 산출

```bash
python explain/explain_shap.py --model models/out/lgb_all_numeric.joblib \
  --x-csv models/out/X_test.csv --out explain/out
```

---

## 🧾 데이터 포맷

### page.meta.json (예)
```json
{
  "url": "http://short.url/abc",
  "final_url": "https://site.com/login",
  "domain": "site.com",
  "is_https": true,
  "status": 200,
  "html_path": "data/collected/<id>/page.html",
  "text_len": 1234,
  "fetched_at": "2025-10-13 02:34:56",
  "index": 17,
  "attempts": 1
}
```

### fusion.csv 주요 열
- 식별/메타: `id`, `url`, `domain`, `is_https`, `status`
- URL 피처: `url_len, hostname_len, path_len, num_digits, num_hyphen, num_underscore, num_dots, has_at, is_ip, is_https, host_entropy, path_depth, has_suspicious_kw, suspicious_token_count`
- HTML 피처: `text_len, num_forms, num_inputs, num_links, num_scripts, num_iframes, has_hidden_styles, has_login_keywords, external_script_count`
- 텍스트 차원축소: `svd_0 ... svd_k`

---

## ⚙️ 옵션 표

| 스크립트 | 핵심 옵션 | 설명 |
|---|---|---|
| `collect_phish.py` | `--limit, --retries, --timeout, --headless, --delay, --user-agent` | 렌더링/메타 수집 |
| `extract_features.py` | `--svd_components, --max_tfidf_features` | TF-IDF SVD 가드, 텍스트 축소 |
| `train_fusion.py` | `--out-dir, --label-file` | 검증 분리, 조기종료, 3종 모델 산출 |
| `explain_shap.py` | `--sample-size` | 요약 플롯/중요도 산출 |

---

## 🔒 재현성 · 안전
- `random_state=42` 고정, 경로는 상대경로 권장
- 도메인 누수 방지: 필요 시 도메인 기반 분할(Group split)
- 메모리: SVD 적극 사용, SHAP `--sample-size` 조절, 플롯 후 `plt.close()`
- 윤리/법규: robots.txt 준수, 민감정보/계정정보 수집 금지

---

## 🛠 트러블슈팅

<details>
<summary><strong>자주 묻는 문제</strong></summary>

- Playwright 실패 → `playwright install` 실행/프록시 확인
- TF-IDF empty vocabulary → 공백 텍스트 여부 확인
- 메모리 부족 → `--svd_components`, `--max_tfidf_features`, `--sample-size` 축소
- 과대평가 의심 → 동일 도메인 학습/검증/테스트 분리 확인

</details>

---

<div align="center">
개선 제안·이슈 환영합니다 🙌
</div>
