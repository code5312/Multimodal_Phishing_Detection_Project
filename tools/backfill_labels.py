import os, json

PHISH_LIST = "urls/phish.txt"
BENIGN_LIST = "urls/benign.txt"
BASE = "data/collected"

def load_set(p):
    s = set()
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if line and not line.startswith("#"):
                s.add(line)
    return s

phish = load_set(PHISH_LIST)
benign = load_set(BENIGN_LIST)

n_upd = 0; n_skip = 0
for d in os.listdir(BASE):
    meta = os.path.join(BASE, d, "page.meta.json")
    if not os.path.exists(meta): 
        continue
    try:
        m = json.load(open(meta, encoding="utf-8"))
        if "label" in m: 
            n_skip += 1
            continue
        u = m.get("url","")
        if u in phish:
            m["label"] = 1
        elif u in benign:
            m["label"] = 0
        else:
            # 모르면 건너뜀
            continue
        json.dump(m, open(meta, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        n_upd += 1
    except Exception:
        pass

print("updated labels:", n_upd, "  skipped(has label):", n_skip)
