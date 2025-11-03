"""
SHAP 기반 모델 설명
훈련된 모델의 예측을 SHAP를 사용하여 설명합니다.
학습된 LightGBM 모델과, 그 모델이 학습에 사용한 특징 열(feature_names) 순서를 불러온 뒤
numeric-only와 멀티모달(텍스트 TF-IDF + 숫자 피처 결합) 두 경우를 한 파일에서 자동으로 처리하는 통합 버전을 만들어줬어.
번들(joblib) 안에 무엇이 들어있느냐에 따라 동작이 달라지며, 아래 키들을 “있으면 사용, 없으면 우회”하도록 설계했어.
SHAP API 변화에 대한 호환 가드 포함, 오탐/미탐 개별 force plot도 선택적으로 생성할 수 있도록 했어.
"""
"""
explain_shap.py
- Loads trained LightGBM model bundle (joblib) with keys: {'model','feature_names'}
- Reads X CSV aligned to the feature_names
- Computes SHAP values on a sampled subset for speed
- Saves summary plot PNG and global importance CSV
Usage:
  python explain_shap.py --model models/output/lgb_all_numeric.joblib --x-csv models/output/X_test.csv --out explain/output
"""
import os, argparse, joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix

def _build_design_matrix(df, bundle, sample_idx=None):
    model = bundle['model']
    has_text = ('vectorizer' in bundle) and ('numeric_cols' in bundle)
    if has_text:
        vectorizer  = bundle['vectorizer']
        numeric_cols = bundle['numeric_cols']
        text_col    = bundle.get('text_column', 'text')
        scaler      = bundle.get('scaler', None)

        X_df = df if sample_idx is None else df.loc[sample_idx]
        if text_col not in X_df.columns:
            raise ValueError(f"Multimodal bundle expects text column '{text_col}' in CSV.")
        texts = X_df[text_col].fillna("").astype(str).tolist()
        X_text = vectorizer.transform(texts)  # sparse

        missing = [c for c in numeric_cols if c not in X_df.columns]
        if missing:
            raise ValueError(f"CSV is missing numeric columns used in training: {missing[:10]} ...")
        X_num = X_df[numeric_cols].fillna(0.0).values

        if scaler is not None:
            try:
                X_num = scaler.transform(X_num)
            except Exception:
                print("[WARN] scaler.transform failed; proceeding without scaling for numeric part.")

        X_num_sp = csr_matrix(X_num)
        X_comb = hstack([X_text, X_num_sp])
        feat_names_text = [f"tfidf_{i}" for i in range(X_text.shape[1])]
        feat_names = feat_names_text + list(numeric_cols)
        return X_comb, feat_names
    else:
        feat_names = bundle['feature_names']
        X_df = df if sample_idx is None else df.loc[sample_idx]
        missing = [c for c in feat_names if c not in X_df.columns]
        if missing:
            raise ValueError(f"X CSV missing required features: {missing[:10]} ...")
        return X_df[feat_names].values, feat_names

def _get_shap_values(explainer, X):
    try:
        shap_obj = explainer.shap_values(X)
        if isinstance(shap_obj, list) and len(shap_obj) == 2:
            return shap_obj[1]
        return shap_obj
    except Exception:
        exp = explainer(X)
        vals = exp.values
        if isinstance(vals, np.ndarray) and vals.ndim == 3 and vals.shape[1] == 2:
            return vals[:, 1, :]
        return vals

def main(model_path, x_csv, out_dir, sample_size=200, topn=30, bg_size=100, preds_csv=None):
    os.makedirs(out_dir, exist_ok=True)
    bundle = joblib.load(model_path)
    model = bundle['model']

    df = pd.read_csv(x_csv)
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("Empty X CSV.")

    sample_idx = df.sample(n=sample_size, random_state=42).index if n_rows > sample_size else df.index
    bg_idx     = df.sample(n=bg_size, random_state=13).index if n_rows > bg_size else df.index

    X_bg, feat_names = _build_design_matrix(df, bundle, sample_idx=bg_idx)
    X_sample, _ = _build_design_matrix(df, bundle, sample_idx=sample_idx)

    def to_dense_if_sparse(X):
        try: return X.toarray()
        except Exception: return X

    X_bg_dense = to_dense_if_sparse(X_bg)
    X_sample_dense = to_dense_if_sparse(X_sample)

    explainer = shap.TreeExplainer(model, data=X_bg_dense, feature_perturbation="interventional")
    shap_matrix = _get_shap_values(explainer, X_sample_dense)

    mean_abs = np.abs(shap_matrix).mean(axis=0)
    imp_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv(os.path.join(out_dir, "global_shap_importance.csv"), index=False)

    topn = min(topn, len(feat_names))
    order = np.argsort(-mean_abs)[:topn]
    feat_top = [feat_names[i] for i in order]
    X_sample_top = pd.DataFrame(X_sample_dense[:, order], columns=feat_top)
    shap_top = shap_matrix[:, order]

    shap.summary_plot(shap_top, X_sample_top, feature_names=feat_top, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"summary_top{topn}.png"), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved SHAP summary (Top-{topn}) and global importance to {out_dir}")

    # optional: per-case force plots using predictions CSV
    if preds_csv and os.path.exists(preds_csv):
        try:
            preds = pd.read_csv(preds_csv)
            if 'id' in preds.columns and 'id' in df.columns:
                sample_ids = set(df.loc[sample_idx, 'id']) if 'id' in df.columns else set()
                miss = preds[(preds['y_true'] != preds['y_pred']) & (preds['id'].isin(sample_ids))].head(5)
                if len(miss) > 0:
                    sample_pos_map = {orig_idx: pos for pos, orig_idx in enumerate(sample_idx)}
                    base_value = explainer.expected_value
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else base_value[0]
                    for _, row in miss.iterrows():
                        orig_idx = df.index[df['id'] == row['id']]
                        if len(orig_idx) == 0: continue
                        orig_idx = orig_idx[0]
                        if orig_idx not in sample_pos_map: continue
                        pos = sample_pos_map[orig_idx]
                        x_row = X_sample_top.iloc[pos:pos+1]
                        sv_row = shap_top[pos]
                        shap.force_plot(base_value=base_value, shap_values=sv_row, features=x_row, matplotlib=True, show=False)
                        plt.savefig(os.path.join(out_dir, f"force_{row['id']}.png"), bbox_inches='tight'); plt.close()
                    print(f"[INFO] Saved force plots for up to {len(miss)} misclassified samples.")
                else:
                    print("[INFO] No misclassified samples (within SHAP sample set). Skipping force plots.")
            else:
                print("[INFO] preds_csv provided but 'id' column not found in both CSVs. Skipping force plots.")
        except Exception as e:
            print(f"[WARN] Could not generate force plots: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to joblib bundle saved by training")
    parser.add_argument("--x-csv", required=True, help="CSV with features and (optionally) a text column used by vectorizer")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--sample-size", type=int, default=200, help="Rows to sample for SHAP explanations")
    parser.add_argument("--topn", type=int, default=30, help="Top-N features to visualize in summary plot")
    parser.add_argument("--bg-size", type=int, default=100, help="Background size for TreeExplainer")
    parser.add_argument("--preds-csv", required=False, help="(Optional) predictions csv (id,y_true,y_pred) for per-case force plots")
    args = parser.parse_args()
    main(args.model, args.x_csv, args.out, args.sample_size, args.topn, args.bg_size, args.preds_csv)
