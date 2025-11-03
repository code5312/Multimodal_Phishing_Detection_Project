"""
퓨전 모델 훈련기
다양한 특성을 결합하여 피싱 사이트 탐지 모델을 훈련합니다.
"""

# -*- coding: utf-8 -*-
"""
train_fusion.py
- 도메인(eTLD+1) 그룹 분할(70/15/15)
- LightGBM: URL-only / HTML-only / All-numeric / Multimodal
- Logistic Regression (텍스트 전용, 대형 TF-IDF 50k~100k)
- 평가: ROC-AUC, PR-AUC, Best-F1 임계값, Precision@Recall>=0.95
"""

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------- utils ----------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def plot_roc_pr(y_true, y_prob, tag, out_dir):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(); plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {tag}")
    plt.savefig(os.path.join(out_dir, f"roc_{tag}.png"), bbox_inches='tight'); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AP={ap:.3f}) - {tag}")
    plt.savefig(os.path.join(out_dir, f"pr_{tag}.png"), bbox_inches='tight'); plt.close()
    return ap

def threshold_for_best_f1(y_true, y_prob):
    ths = np.linspace(0.05, 0.95, 19)
    scores = [f1_score(y_true, (y_prob >= t).astype(int)) for t in ths]
    return float(ths[int(np.argmax(scores))])

def threshold_for_recall(y_true, y_prob, target=0.95):
    precisions, recalls, ths = precision_recall_curve(y_true, y_prob)
    # recalls 길이는 ths+1 이므로 마지막에 1.0 하나 보강
    for r, t in zip(recalls, np.append(ths, 1.0)):
        if r >= target:
            return float(t)
    return 0.5

def group_split_70_15_15(df, y, group_col="domain", random_state=42):
    """eTLD+1 domain을 그룹으로 70/15/15 분할(누수 방지)."""
    groups = df[group_col].fillna("__NA__").astype(str)

    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state)
    tr_idx, tmp_idx = next(gss1.split(df, y, groups))

    df_tmp = df.iloc[tmp_idx].reset_index(drop=True)
    y_tmp = y.iloc[tmp_idx].reset_index(drop=True)
    groups_tmp = df_tmp[group_col].fillna("__NA__").astype(str)

    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=random_state+1)
    va_rel, te_rel = next(gss2.split(df_tmp, y_tmp, groups_tmp))
    va_idx = df_tmp.index[va_rel]
    te_idx = df_tmp.index[te_rel]

    # 원래 df의 절대 인덱스로 환산
    va_idx = df.iloc[tmp_idx].index[va_idx]
    te_idx = df.iloc[tmp_idx].index[te_idx]
    return tr_idx, va_idx, te_idx


# ---------------------- LightGBM common ----------------------
def train_eval_save_lgb(model_name, clf, Xtr, ytr, Xva, yva, Xte, yte, out_dir,
                        feature_cols=None, bundle_extra=None, random_state=42):
    # 클래스 불균형 자동 보정
    pos = int((ytr == 1).sum()); neg = int((ytr == 0).sum())
    spw = max(1.0, (neg / max(1, pos)))
    clf.set_params(scale_pos_weight=spw, random_state=random_state, n_jobs=-1)

    clf.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric='logloss',
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=0)]
    )

    # 검증 F1 최적 임계값
    y_val_prob = clf.predict_proba(Xva)[:, 1]
    best_t = threshold_for_best_f1(yva, y_val_prob)

    # 테스트 성능
    y_prob = clf.predict_proba(Xte)[:, 1]
    y_pred = (y_prob >= best_t).astype(int)
    auc = roc_auc_score(yte, y_prob)
    ap = plot_roc_pr(yte, y_prob, model_name, out_dir)
    report = classification_report(yte, y_pred, digits=4)

    # R@0.95 기준
    thr_r095 = threshold_for_recall(yva, y_val_prob, 0.95)
    y_pred_r095 = (y_prob >= thr_r095).astype(int)
    prec_r095 = precision_score(yte, y_pred_r095, zero_division=0)
    rec_r095  = recall_score(yte, y_pred_r095, zero_division=0)

    # 혼동행렬
    cm = confusion_matrix(yte, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{model_name}.png"), bbox_inches='tight'); plt.close()

    # 중요도 (가능한 경우)
    if feature_cols is not None:
        imp = pd.DataFrame({
            "feature": feature_cols,
            "gain_importance": clf.booster_.feature_importance(importance_type="gain")
        }).sort_values("gain_importance", ascending=False)
        imp.to_csv(os.path.join(out_dir, f"feature_importance_{model_name}.csv"), index=False)

    # 리포트 저장
    with open(os.path.join(out_dir, f"report_{model_name}.txt"), "w", encoding="utf-8") as f:
        f.write(f"AUC: {auc:.6f}\nAP: {ap:.6f}\n")
        f.write(f"Best threshold (F1 on val): {best_t:.3f}\n")
        f.write(f"Precision@Recall>=0.95: {prec_r095:.4f} (thr={thr_r095:.3f}, recall={rec_r095:.4f})\n\n")
        f.write(report)

    # 번들 저장
    bundle = {'model': clf, 'random_state': random_state,
              'lgb_version': lgb.__version__, 'threshold': float(best_t)}
    if feature_cols is not None: bundle['feature_names'] = list(feature_cols)
    if bundle_extra: bundle.update(bundle_extra)
    joblib.dump(bundle, os.path.join(out_dir, f"lgb_{model_name}.joblib"))


# ---------------------- main ----------------------
def main(input_csv, label_csv, out_dir,
         text_column="text",
         tfidf_max_features=5000,           # 멀티모달 텍스트 파트 (작게)
         lr_tfidf_max_features=80000,       # LR 전용 대형 사전
         lr_C_grid="0.01,0.03,0.1,0.3,1,3,10",
         random_state=42):

    ensure_dir(out_dir)
    np.random.seed(random_state)

    # ---------- 데이터 로드 ----------
    df = pd.read_csv(input_csv)
    if label_csv and os.path.exists(label_csv):
        labels = pd.read_csv(label_csv)
        df = df.merge(labels, on="id", how="left")
    elif 'label' not in df.columns:
        raise ValueError("Label file not provided and no label column in fusion csv")

    df = df.dropna(subset=['label']).reset_index(drop=True)
    y = df['label'].astype(int)

    # ---------- 수치 피처 매트릭스 ----------
    drop_cols = ['id', 'url', 'label', 'html', text_column]
    X_all = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X_all = X_all.select_dtypes(include=[np.number]).fillna(0)

    # 그룹 분할(도메인 기준)
    tr_idx, va_idx, te_idx = group_split_70_15_15(df, y, group_col="domain", random_state=random_state)

    X_train_num = X_all.loc[tr_idx].reset_index(drop=True)
    X_val_num   = X_all.loc[va_idx].reset_index(drop=True)
    X_test_num  = X_all.loc[te_idx].reset_index(drop=True)

    y_train = y.loc[tr_idx].reset_index(drop=True)
    y_val   = y.loc[va_idx].reset_index(drop=True)
    y_test  = y.loc[te_idx].reset_index(drop=True)

    ids_test = df.loc[te_idx, 'id']

    # ---------- 피처 그룹 ----------
    cand_url = [
        'url_len','hostname_len','path_len','num_digits','num_hyphen','num_underscore','num_dots',
        'has_at','is_ip','is_https','host_entropy','path_depth','has_suspicious_kw','suspicious_token_count'
    ]
    cand_html = [
        'text_len','num_forms','num_inputs','num_links','num_scripts','num_iframes',
        'has_hidden_styles','has_login_keywords','external_script_count'
    ]
    url_cols  = [c for c in cand_url  if c in X_all.columns]
    html_cols = [c for c in cand_html if c in X_all.columns]
    all_cols  = list(X_all.columns)

    # ---------- LightGBM 모델들 ----------
    lgb_params = dict(
        n_estimators=2000, learning_rate=0.05,
        max_depth=-1, num_leaves=512, subsample=0.9, colsample_bytree=0.9
    )

    # 1) URL-only
    if url_cols:
        clf = lgb.LGBMClassifier(**lgb_params)
        train_eval_save_lgb("url_only", clf,
                            X_train_num[url_cols], y_train,
                            X_val_num[url_cols], y_val,
                            X_test_num[url_cols], y_test,
                            out_dir, feature_cols=url_cols, random_state=random_state)

    # 2) HTML-only
    if html_cols:
        clf = lgb.LGBMClassifier(**lgb_params)
        train_eval_save_lgb("html_only", clf,
                            X_train_num[html_cols], y_train,
                            X_val_num[html_cols], y_val,
                            X_test_num[html_cols], y_test,
                            out_dir, feature_cols=html_cols, random_state=random_state)

    # 3) All numeric
    if all_cols:
        clf = lgb.LGBMClassifier(**lgb_params)
        train_eval_save_lgb("all_numeric", clf,
                            X_train_num[all_cols], y_train,
                            X_val_num[all_cols], y_val,
                            X_test_num[all_cols], y_test,
                            out_dir, feature_cols=all_cols, random_state=random_state)

    # 4) Multimodal (텍스트 TF-IDF(작게) + 수치)
    if text_column in df.columns:
        tr_texts = df.loc[tr_idx, text_column].fillna("").astype(str).tolist()
        va_texts = df.loc[va_idx, text_column].fillna("").astype(str).tolist()
        te_texts = df.loc[te_idx, text_column].fillna("").astype(str).tolist()

        vec_mm = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1, 2))
        Xtr_text = vec_mm.fit_transform(tr_texts)
        Xva_text = vec_mm.transform(va_texts)
        Xte_text = vec_mm.transform(te_texts)

        scaler = StandardScaler(with_mean=False)
        Xtr_num_sc = csr_matrix(scaler.fit_transform(X_train_num[all_cols].values))
        Xva_num_sc = csr_matrix(scaler.transform(X_val_num[all_cols].values))
        Xte_num_sc = csr_matrix(scaler.transform(X_test_num[all_cols].values))

        Xtr_mm = hstack([Xtr_text, Xtr_num_sc]).tocsr()
        Xva_mm = hstack([Xva_text, Xva_num_sc]).tocsr()
        Xte_mm = hstack([Xte_text, Xte_num_sc]).tocsr()

        clf = lgb.LGBMClassifier(**lgb_params)
        train_eval_save_lgb(
            "multimodal", clf,
            Xtr_mm, y_train, Xva_mm, y_val, Xte_mm, y_test,
            out_dir, feature_cols=None,  # 중요도 CSV는 수치 전용에서만 저장
            bundle_extra={
                'vectorizer': vec_mm,
                'scaler': scaler,
                'numeric_cols': all_cols,
                'text_column': text_column
            },
            random_state=random_state
        )
    else:
        print(f"[INFO] '{text_column}' column not found. Skip multimodal.")

    # 5) Logistic Regression baseline (텍스트 전용, 대형 사전)
    if text_column in df.columns:
        tr_texts = df.loc[tr_idx, text_column].fillna("").astype(str).tolist()
        va_texts = df.loc[va_idx, text_column].fillna("").astype(str).tolist()
        te_texts = df.loc[te_idx, text_column].fillna("").astype(str).tolist()

        vec_lr = TfidfVectorizer(max_features=lr_tfidf_max_features, ngram_range=(1, 2), min_df=2)
        Xtr = vec_lr.fit_transform(tr_texts)
        Xva = vec_lr.transform(va_texts)
        Xte = vec_lr.transform(te_texts)

        C_list = [float(x) for x in lr_C_grid.split(",")]

        for C in C_list:
            tag = f"lr_text_C{str(C).replace('.', 'p')}"
            lr = LogisticRegression(
                solver="saga", penalty="l2", C=C,
                class_weight="balanced", max_iter=5000, n_jobs=-1
            )
            lr.fit(Xtr, y_train)

            y_val_prob = lr.predict_proba(Xva)[:, 1]
            y_test_prob = lr.predict_proba(Xte)[:, 1]

            # R@0.95 임계값
            thr_r095 = threshold_for_recall(y_val, y_val_prob, 0.95)
            y_pred_r095 = (y_test_prob >= thr_r095).astype(int)
            prec_r095 = precision_score(y_test, y_pred_r095, zero_division=0)
            rec_r095  = recall_score(y_test, y_pred_r095, zero_division=0)

            # F1 기준 임계값(참고)
            thr_f1 = threshold_for_best_f1(y_val, y_val_prob)
            y_pred_f1 = (y_test_prob >= thr_f1).astype(int)

            auc = roc_auc_score(y_test, y_test_prob)
            ap  = plot_roc_pr(y_test, y_test_prob, tag, out_dir)
            rep = classification_report(y_test, y_pred_f1, digits=4)

            with open(os.path.join(out_dir, f"report_{tag}.txt"), "w", encoding="utf-8") as f:
                f.write(f"AUC: {auc:.6f}\nAP: {ap:.6f}\n")
                f.write(f"Precision@Recall>=0.95: {prec_r095:.4f} (thr={thr_r095:.3f}, recall={rec_r095:.4f})\n")
                f.write(f"Best-F1 threshold on val: {thr_f1:.3f}\n\n{rep}")

            joblib.dump({'model': lr, 'vectorizer': vec_lr,
                         'threshold_r095': float(thr_r095),
                         'threshold_f1': float(thr_f1)},
                        os.path.join(out_dir, f"lr_{tag}.joblib"))
    else:
        print(f"[INFO] '{text_column}' column not found. Skip LR baseline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="fusion csv")
    parser.add_argument("--label-file", required=False, help="csv with id,label columns")
    parser.add_argument("--out-dir", required=True, help="output dir for models and artifacts")
    parser.add_argument("--text-column", default="text", help="text column name in fusion csv")

    # 멀티모달 텍스트 파트(작은 사전) & LR 전용(큰 사전) 별도 설정
    parser.add_argument("--tfidf-max-features", type=int, default=5000,
                        help="TF-IDF max features for multimodal text part")
    parser.add_argument("--lr-tfidf-max-features", type=int, default=80000,
                        help="TF-IDF max features for LR text baseline (50k~100k 권장)")
    parser.add_argument("--lr-C-grid", default="0.01,0.03,0.1,0.3,1,3,10",
                        help="comma-separated C values for Logistic Regression")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(args.input, args.label_file, args.out_dir,
         args.text_column, args.tfidf_max_features,
         args.lr_tfidf_max_features, args.lr_C_grid, args.random_state)
