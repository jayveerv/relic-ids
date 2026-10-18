"""
AI-based IDS experiments for: "AI-Based Intrusion Detection Systems for Cloud Environments"
Author: Jayveer Vasava (M.Tech Cyber Security, Parul University)

Datasets : NSL-KDD (KDDTrain+ / KDDTest+), UNSW-NB15 (official training / testing sets)
Models   : Logistic Regression, Decision Tree, Random Forest, XGBoost, MLP (DNN)
Studies  : (1) binary detection on official splits, 3 seeds
           (2) multi-class detection
           (3) feature selection (mutual information, top-k) vs all features
           (4) random-split vs official-split evaluation gap
           (5) SHAP explanation of XGBoost
Run      : python run_experiments.py  (expects ../data, writes ../results)
"""
import json, os, time, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RELIC_DATA", os.path.join(HERE, "..", "..", "data"))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]

# ---------------------------------------------------------------- data
KDD_COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
 "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
 "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
 "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
 "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
 "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
 "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
KDD_CAT = {  # standard NSL-KDD attack -> category mapping (incl. test-only attacks)
 "normal":"Normal",
 **{a:"DoS" for a in ["back","land","neptune","pod","smurf","teardrop","apache2","mailbomb","processtable","udpstorm"]},
 **{a:"Probe" for a in ["ipsweep","nmap","portsweep","satan","mscan","saint"]},
 **{a:"R2L" for a in ["ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster",
                     "sendmail","named","snmpgetattack","snmpguess","xlock","xsnoop","worm"]},
 **{a:"U2R" for a in ["buffer_overflow","loadmodule","perl","rootkit","httptunnel","ps","sqlattack","xterm"]},
}

def load_nslkdd():
    tr = pd.read_csv(os.path.join(DATA, "KDDTrain+.txt"), names=KDD_COLS)
    te = pd.read_csv(os.path.join(DATA, "KDDTest+.txt"), names=KDD_COLS)
    for d in (tr, te):
        d["cat"] = d["label"].map(KDD_CAT)
        d["y"] = (d["label"] != "normal").astype(int)
    assert tr["cat"].notna().all() and te["cat"].notna().all(), "unmapped attack label"
    drop = ["label", "difficulty", "cat", "y"]
    return dict(name="NSL-KDD", Xtr=tr.drop(columns=drop), Xte=te.drop(columns=drop),
                ytr=tr["y"].values, yte=te["y"].values, ctr=tr["cat"].values, cte=te["cat"].values,
                cat_cols=["protocol_type", "service", "flag"])

def load_unsw():
    tr = pd.read_csv(os.path.join(DATA, "UNSW_NB15_training-set.csv"), encoding="utf-8-sig")
    te = pd.read_csv(os.path.join(DATA, "UNSW_NB15_testing-set.csv"), encoding="utf-8-sig")
    drop = ["id", "attack_cat", "label"]
    return dict(name="UNSW-NB15", Xtr=tr.drop(columns=drop), Xte=te.drop(columns=drop),
                ytr=tr["label"].values, yte=te["label"].values,
                ctr=tr["attack_cat"].str.strip().values, cte=te["attack_cat"].str.strip().values,
                cat_cols=["proto", "service", "state"])

def preprocess(Xtr, Xte, cat_cols):
    num = [c for c in Xtr.columns if c not in cat_cols]
    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                            ("num", StandardScaler(), num)])
    A, B = ct.fit_transform(Xtr), ct.transform(Xte)
    names = [n.split("__", 1)[1] for n in ct.get_feature_names_out()]
    return A.astype(np.float32), B.astype(np.float32), names

def models(seed, n_classes=2):
    xgb_obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, objective=xgb_obj,
                                 tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0),
        "MLP (DNN)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, early_stopping=True,
                                   random_state=seed),
    }

def evaluate(m, Xtr, ytr, Xte, yte):
    """Binary evaluation: fit, predict, and report detection + cost metrics."""
    t0 = time.perf_counter(); m.fit(Xtr, ytr); ttrain = time.perf_counter() - t0
    t0 = time.perf_counter(); p = m.predict(Xte); tinf = time.perf_counter() - t0
    prob = m.predict_proba(Xte)[:, 1]
    tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
    r = dict(accuracy=accuracy_score(yte, p), precision=precision_score(yte, p), recall=recall_score(yte, p),
             f1=f1_score(yte, p), fpr=fp / (fp + tn), auc=roc_auc_score(yte, prob),
             cm=[int(tn), int(fp), int(fn), int(tp)], train_s=ttrain,
             infer_us_per_sample=1e6 * tinf / len(yte))
    return r, p

def main():
    results = {"binary": [], "multiclass": [], "feature_selection": [], "split_gap": [], "dataset_stats": {}}
    for D in (load_nslkdd(), load_unsw()):
        name = D["name"]; print(f"\n===== {name} =====")
        Xtr, Xte, feat = preprocess(D["Xtr"], D["Xte"], D["cat_cols"])
        results["dataset_stats"][name] = dict(
            train_rows=len(D["ytr"]), test_rows=len(D["yte"]), raw_features=D["Xtr"].shape[1],
            encoded_features=Xtr.shape[1], train_attack_ratio=float(D["ytr"].mean()),
            test_attack_ratio=float(D["yte"].mean()),
            train_classes=pd.Series(D["ctr"]).value_counts().to_dict(),
            test_classes=pd.Series(D["cte"]).value_counts().to_dict())

        # (1) binary, official split, 3 seeds
        for seed in SEEDS:
            for mname, m in models(seed).items():
                r, _ = evaluate(m, Xtr, D["ytr"], Xte, D["yte"])
                r.update(dataset=name, model=mname, seed=seed); results["binary"].append(r)
                print(f"[bin s{seed}] {mname:20s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} fpr={r['fpr']:.4f}")

        # (2) multi-class, seed 0
        le = LabelEncoder().fit(np.concatenate([D["ctr"], D["cte"]]))
        ctr, cte = le.transform(D["ctr"]), le.transform(D["cte"])
        for mname, m in models(0, n_classes=len(le.classes_)).items():
            t0 = time.perf_counter(); m.fit(Xtr, ctr); tt = time.perf_counter() - t0
            p = m.predict(Xte)
            rec = recall_score(cte, p, average=None, labels=range(len(le.classes_)), zero_division=0)
            results["multiclass"].append(dict(dataset=name, model=mname, accuracy=accuracy_score(cte, p),
                macro_f1=f1_score(cte, p, average="macro", zero_division=0),
                weighted_f1=f1_score(cte, p, average="weighted", zero_division=0), train_s=tt,
                per_class_recall=dict(zip(le.classes_.tolist(), rec.tolist()))))
            print(f"[multi] {mname:20s} acc={results['multiclass'][-1]['accuracy']:.4f} "
                  f"macroF1={results['multiclass'][-1]['macro_f1']:.4f}")

        # (3) feature selection: mutual information on train only
        mi = mutual_info_classif(Xtr, D["ytr"], random_state=0)
        order = np.argsort(mi)[::-1]
        results.setdefault("mi_top", {})[name] = [(feat[i], float(mi[i])) for i in order[:15]]
        for k in (10, 20, 30, Xtr.shape[1]):
            idx = order[:k]
            for mname in ("Random Forest", "XGBoost"):
                r, _ = evaluate(models(0)[mname], Xtr[:, idx], D["ytr"], Xte[:, idx], D["yte"])
                r.update(dataset=name, model=mname, k=int(k)); results["feature_selection"].append(r)
                print(f"[fs k={k}] {mname:14s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} train={r['train_s']:.2f}s")

        # (4) random 80/20 split of the TRAIN file vs official test file
        a, b, ya, yb = train_test_split(Xtr, D["ytr"], test_size=0.2, stratify=D["ytr"], random_state=0)
        for mname in ("Decision Tree", "Random Forest", "XGBoost"):
            r_rand, _ = evaluate(models(0)[mname], a, ya, b, yb)
            r_off, _ = evaluate(models(0)[mname], Xtr, D["ytr"], Xte, D["yte"])
            results["split_gap"].append(dict(dataset=name, model=mname, random_split_acc=r_rand["accuracy"],
                random_split_f1=r_rand["f1"], official_acc=r_off["accuracy"], official_f1=r_off["f1"]))
            print(f"[gap] {mname:14s} random={r_rand['accuracy']:.4f} official={r_off['accuracy']:.4f}")

        # (5) SHAP on XGBoost (binary, seed 0)
        import shap
        xgb = models(0)["XGBoost"].fit(Xtr, D["ytr"])
        rng = np.random.RandomState(0)
        sample = Xte[rng.choice(len(Xte), 2000, replace=False)]
        sv = shap.TreeExplainer(xgb).shap_values(sample)
        imp = np.abs(sv).mean(0); o = np.argsort(imp)[::-1][:10]
        results.setdefault("shap_top10", {})[name] = [(feat[i], float(imp[i])) for i in o]
        np.save(os.path.join(OUT, f"shap_{name}.npy"), sv); np.save(os.path.join(OUT, f"shap_X_{name}.npy"), sample)
        json.dump(feat, open(os.path.join(OUT, f"features_{name}.json"), "w"))

    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    print("\nSaved", os.path.join(OUT, "results.json"))

if __name__ == "__main__":
    main()
