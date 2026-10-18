# Data

Download the four files into this folder (not redistributed here):

| File | Source |
|---|---|
| KDDTrain+.txt, KDDTest+.txt | NSL-KDD (Tavallaee et al., 2009). Mirror used: https://github.com/defcom17/NSL_KDD |
| UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv | UNSW-NB15 official partition (Moustafa and Slay, 2015). https://research.unsw.edu.au/projects/unsw-nb15-dataset |

```bash
B=https://raw.githubusercontent.com/defcom17/NSL_KDD/master
curl -L -o KDDTrain+.txt "$B/KDDTrain%2B.txt"
curl -L -o KDDTest+.txt  "$B/KDDTest%2B.txt"
```
