"""
Train the models the app uses and save them to models/.

  full : all 42 UNSW-NB15 features, for CSV files in UNSW-NB15 format
  live : features the packet extractor can compute reliably, excluding testbed artefacts
         (sttl, dttl, ct_state_ttl, swin, dwin, stcpb, dtcpb). Used for PCAP files and live capture.

Each model = binary XGBoost (attack / normal) + 10-class XGBoost (attack type), trained on the
official UNSW-NB15 training partition and scored on the official testing partition.

Run: python train_models.py --data ../ids/data
"""
import argparse, json, os, time
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

from relic.features import FEATURE_COLS

CAT = ["proto", "service", "state"]
ARTEFACTS = ["sttl", "dttl", "ct_state_ttl", "swin", "dwin", "stcpb", "dtcpb"]
SETS = {"full": FEATURE_COLS, "live": [c for c in FEATURE_COLS if c not in ARTEFACTS]}


def xgb(n_classes, seed=0):
    return XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.1, tree_method="hist", n_jobs=-1,
                         random_state=seed, verbosity=0,
                         objective="binary:logistic" if n_classes == 2 else "multi:softprob")


def main(data):
    tr = pd.read_csv(os.path.join(data, "UNSW_NB15_training-set.csv"), encoding="utf-8-sig")
    te = pd.read_csv(os.path.join(data, "UNSW_NB15_testing-set.csv"), encoding="utf-8-sig")
    for d in (tr, te):
        d["attack_cat"] = d["attack_cat"].str.strip()
    os.makedirs("models", exist_ok=True)
    report = {}
    for name, feats in SETS.items():
        t0 = time.time()
        cat = [c for c in CAT if c in feats]
        num = [c for c in feats if c not in cat]
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                                 ("num", StandardScaler(), num)])
        Xtr = pre.fit_transform(tr[feats]).astype(np.float32)
        Xte = pre.transform(te[feats]).astype(np.float32)
        enc_names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]

        # binary
        binm = xgb(2).fit(Xtr, tr["label"])
        prob = binm.predict_proba(Xte)[:, 1]
        # operating points from a validation split of the training data
        a, v, ya, yv = train_test_split(Xtr, tr["label"], test_size=0.2, stratify=tr["label"], random_state=0)
        pv = xgb(2).fit(a, ya).predict_proba(v)[:, 1]
        neg = np.sort(pv[yv.values == 0])
        presets = {"Balanced (0.50)": 0.5,
                   "Low false alarms (val FPR 1%)": float(np.quantile(neg, 0.99)),
                   "High sensitivity (0.20)": 0.2}
        p = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(te["label"], p).ravel()
        # attack type
        le = LabelEncoder().fit(pd.concat([tr["attack_cat"], te["attack_cat"]]))
        mcm = xgb(len(le.classes_)).fit(Xtr, le.transform(tr["attack_cat"]))
        mp = mcm.predict(Xte)
        yte_c = le.transform(te["attack_cat"])
        metrics = dict(
            accuracy=accuracy_score(te["label"], p), precision=precision_score(te["label"], p),
            recall=recall_score(te["label"], p), f1=f1_score(te["label"], p), fpr=fp / (fp + tn),
            auc=roc_auc_score(te["label"], prob),
            type_accuracy=accuracy_score(yte_c, mp), type_macro_f1=f1_score(yte_c, mp, average="macro"),
            per_class_recall=dict(zip(le.classes_.tolist(),
                                      recall_score(yte_c, mp, average=None, labels=range(len(le.classes_))).tolist())),
            n_train=len(tr), n_test=len(te), n_features=len(feats), train_seconds=time.time() - t0)
        benign_median = tr.loc[tr["label"] == 0, num].median().to_dict()
        bundle = dict(name=name, features=feats, cat=cat, num=num, pre=pre, enc_names=enc_names,
                      binary=binm, multiclass=mcm, classes=le.classes_.tolist(), presets=presets,
                      metrics=metrics, benign_median=benign_median)
        joblib.dump(bundle, os.path.join("models", f"relic_{name}.joblib"), compress=3)
        report[name] = {k: v for k, v in metrics.items()} | {"presets": presets}
        print(f"[{name}] features={len(feats)} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
              f"fpr={metrics['fpr']:.4f} auc={metrics['auc']:.4f} type_acc={metrics['type_accuracy']:.4f}")
    json.dump(report, open(os.path.join("models", "model_card.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data")
    main(ap.parse_args().data)
