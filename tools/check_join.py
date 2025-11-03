# tools/check_join.py
import pandas as pd
import tldextract as tld

f = pd.read_csv("data/fusion.csv")
y = pd.read_csv("data/labels.csv")

df = f.merge(y, on="id", how="inner")
print("fusion rows:", len(f))
print("labels rows:", len(y))
print("joined rows:", len(df))

# 숫자 피처 개수
X = df.drop(columns=["id","url","label","html"], errors="ignore").select_dtypes("number").fillna(0)
print("numeric features:", X.shape[1])

# 클래스 분포
yc = df["label"].astype(int).value_counts().to_dict()
print("class balance:", {0: yc.get(0,0), 1: yc.get(1,0)})

# eTLD+1(도메인 그룹) 개수
def etld1(u: str) -> str:
    e = tld.extract(u if isinstance(u,str) else "")
    return (e.domain + "." + e.suffix).strip(".")
g = df["url"].apply(etld1)
print("unique domain groups:", g.nunique())
