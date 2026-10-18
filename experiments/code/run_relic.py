"""
RELIC-IDS: Reliability Evaluation for Learning-based Intrusion detection in Cloud
Extension experiments for the paper "Beyond 99% Accuracy".

R1  Zero-day test        : NSL-KDD test attacks split into types seen / unseen in training
R2  False-alarm control  : decision threshold chosen on a validation split for target FPR (5%, 1%)
R3  Rare-class balancing : multi-class with no weighting vs balanced sample weights vs SMOTE
R4  Adversarial mimicry  : top-SHAP features of detected attacks moved toward the benign median
R5  Deployment cost      : model size, single-flow latency, batch throughput (1 thread)

Run: python run_relic.py   (expects ../data, writes ../results/relic.json)
"""
import json, os, pickle, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from run_experiments import load_nslkdd, load_unsw, preprocess, models, KDD_COLS, KDD_CAT, DATA, OUT

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2]
R = {}

def fpr_rec(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return dict(fpr=fp / (fp + tn), recall=tp / (tp + fn), precision=tp / max(tp + fp, 1), f1=f1_score(y, p))

def fit_preproc(D):
    """Return fitted ColumnTransformer-equivalent via preprocess() plus a raw->encoded function."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num = [c for c in D["Xtr"].columns if c not in D["cat_cols"]]
    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), D["cat_cols"]),
                            ("num", StandardScaler(), num)]).fit(D["Xtr"])
    names = [n.split("__", 1)[1] for n in ct.get_feature_names_out()]
    return ct, (lambda X: ct.transform(X).astype(np.float32)), names

# ======================================================================= R1 zero-day
def r1_zero_day():
    D = load_nslkdd()
    tr = pd.read_csv(os.path.join(DATA, "KDDTrain+.txt"), names=KDD_COLS)
    te = pd.read_csv(os.path.join(DATA, "KDDTest+.txt"), names=KDD_COLS)
    seen = set(tr.label.unique())
    attack = te.label != "normal"
    novel = attack & ~te.label.isin(seen)
    known = attack & te.label.isin(seen)
    Xtr, Xte, _ = preprocess(D["Xtr"], D["Xte"], D["cat_cols"])
    out = dict(n_known=int(known.sum()), n_novel=int(novel.sum()),
               novel_types=sorted(te.label[novel].unique().tolist()), models=[], per_type={})
    print(f"[R1] known attacks={known.sum()}  novel attacks={novel.sum()}  novel types={len(out['novel_types'])}")
    for seed in SEEDS:
        for name, m in models(seed).items():
            m.fit(Xtr, D["ytr"]); p = m.predict(Xte)
            out["models"].append(dict(model=name, seed=seed,
                known_dr=float(p[known.values].mean()), novel_dr=float(p[novel.values].mean()),
                normal_fpr=float(p[(~attack).values].mean())))
            if seed == 0 and name == "XGBoost":
                for t in out["novel_types"]:
                    msk = (te.label == t).values
                    out["per_type"][t] = dict(n=int(msk.sum()), detected=float(p[msk].mean()),
                                              category=KDD_CAT[t])
    df = pd.DataFrame(out["models"]).groupby("model")[["known_dr", "novel_dr"]].mean()
    print(df.round(4))
    R["zero_day"] = out

# ======================================================================= R2 threshold
def r2_threshold():
    out = []
    for D in (load_nslkdd(), load_unsw()):
        Xtr, Xte, _ = preprocess(D["Xtr"], D["Xte"], D["cat_cols"])
        a, v, ya, yv = train_test_split(Xtr, D["ytr"], test_size=0.2, stratify=D["ytr"], random_state=0)
        for name in ("Random Forest", "XGBoost"):
            m = models(0)[name].fit(a, ya)
            pv, pt = m.predict_proba(v)[:, 1], m.predict_proba(Xte)[:, 1]
            neg = np.sort(pv[yv == 0])
            for target in (None, 0.05, 0.01):
                thr = 0.5 if target is None else float(np.quantile(neg, 1 - target))
                r = fpr_rec(D["yte"], (pt >= thr).astype(int))
                r.update(dataset=D["name"], model=name, target_fpr=target, threshold=thr,
                         val_fpr=float((pv[yv == 0] >= thr).mean()))
                out.append(r)
                print(f"[R2] {D['name']:9s} {name:13s} target={target} thr={thr:.4f} "
                      f"testFPR={r['fpr']:.4f} recall={r['recall']:.4f} f1={r['f1']:.4f}")
    R["threshold"] = out

# ======================================================================= R3 rare classes
def r3_rare():
    out = []
    for D in (load_nslkdd(), load_unsw()):
        Xtr, Xte, _ = preprocess(D["Xtr"], D["Xte"], D["cat_cols"])
        le = LabelEncoder().fit(np.concatenate([D["ctr"], D["cte"]]))
        ytr, yte = le.transform(D["ctr"]), le.transform(D["cte"])
        variants = {"None": (Xtr, ytr, None),
                    "Balanced weights": (Xtr, ytr, compute_sample_weight("balanced", ytr))}
        sm = SMOTE(random_state=0, k_neighbors=5)
        Xs, ys = sm.fit_resample(Xtr, ytr)
        variants["SMOTE"] = (Xs, ys, None)
        for vname, (X, y, w) in variants.items():
            for name in ("Random Forest", "XGBoost"):
                m = models(0, n_classes=len(le.classes_))[name]
                m.fit(X, y, sample_weight=w)
                p = m.predict(Xte)
                rec = recall_score(yte, p, average=None, labels=range(len(le.classes_)), zero_division=0)
                out.append(dict(dataset=D["name"], model=name, balancing=vname,
                                accuracy=accuracy_score(yte, p),
                                macro_f1=f1_score(yte, p, average="macro", zero_division=0),
                                per_class_recall=dict(zip(le.classes_.tolist(), rec.tolist()))))
                print(f"[R3] {D['name']:9s} {name:13s} {vname:17s} macroF1={out[-1]['macro_f1']:.4f}")
    R["rare"] = out

# ======================================================================= R4 adversarial mimicry
def r4_adversarial():
    import shap
    out = []
    for D in (load_nslkdd(), load_unsw()):
        ct, enc, names = fit_preproc(D)
        Xtr = enc(D["Xtr"])
        m = models(0)["XGBoost"].fit(Xtr, D["ytr"])
        # top numeric features by mean |SHAP| on a test sample (one-hot columns excluded: not continuous)
        rng = np.random.RandomState(0)
        idx = rng.choice(len(D["Xte"]), 2000, replace=False)
        sv = shap.TreeExplainer(m).shap_values(enc(D["Xte"].iloc[idx]))
        imp = pd.Series(np.abs(sv).mean(0), index=names)
        raw_num = [c for c in D["Xtr"].columns if c not in D["cat_cols"]]
        top = [n for n in imp.sort_values(ascending=False).index if n in raw_num][:3]
        benign_med = D["Xtr"].loc[D["ytr"] == 0, top].median()
        Xte_att = D["Xte"][D["yte"] == 1]
        det = m.predict(enc(Xte_att)) == 1
        base = Xte_att[det]                      # attacks the model currently catches
        for k in (1, 2, 3):
            feats = top[:k]
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                Xa = base.copy()
                for f in feats:
                    Xa[f] = (1 - alpha) * Xa[f] + alpha * benign_med[f]
                    if pd.api.types.is_integer_dtype(D["Xte"][f]):
                        Xa[f] = Xa[f].round().astype(D["Xte"][f].dtype)
                still = float((m.predict(enc(Xa)) == 1).mean())
                out.append(dict(dataset=D["name"], k=k, features=feats, alpha=alpha,
                                n_attacks=int(len(base)), detection_rate=still, evasion_rate=1 - still))
                print(f"[R4] {D['name']:9s} k={k} {feats} alpha={alpha:.2f} detection={still:.4f}")
    R["adversarial"] = out

# ======================================================================= R5 deployment cost
def r5_cost():
    D = load_unsw()
    Xtr, Xte, _ = preprocess(D["Xtr"], D["Xte"], D["cat_cols"])
    out = []
    for name, m in models(0).items():
        if hasattr(m, "n_jobs"): m.set_params(n_jobs=1)
        m.fit(Xtr, D["ytr"])
        size = len(pickle.dumps(m))
        one = Xte[:1]
        for _ in range(20): m.predict(one)
        t0 = time.perf_counter()
        for i in range(500): m.predict(Xte[i:i + 1])
        lat_ms = (time.perf_counter() - t0) / 500 * 1e3
        batch = Xte[:50000]
        t0 = time.perf_counter(); m.predict(batch); dt = time.perf_counter() - t0
        out.append(dict(model=name, size_kb=size / 1024, latency_ms=lat_ms, throughput_fps=len(batch) / dt))
        print(f"[R5] {name:20s} size={size/1024:9.1f} KB  latency={lat_ms:.3f} ms  throughput={len(batch)/dt:,.0f} flows/s")
    R["cost"] = out

if __name__ == "__main__":
    for f in (r1_zero_day, r2_threshold, r3_rare, r4_adversarial, r5_cost):
        f(); json.dump(R, open(os.path.join(OUT, "relic.json"), "w"), indent=1, default=str)
    print("Saved", os.path.join(OUT, "relic.json"))
