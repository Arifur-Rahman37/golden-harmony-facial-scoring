"""
rescale_propsym_to_1to10.py
PropSym P1-P4 scores in the old PropSym files were on a 0-10 scale (10*u, u in [0,1]).
The paper (Eq. 2a) and Combined_technical.csv use 1 + 9*u. This script converts
P1-P4:  new = 1 + 0.9 * old,  then recomputes the naive score (mean of the 10 features),
the percent score, AI-vs-Human-Diff, and the subject-level means.
Usage: put PropSym_image_level_FIXED498.csv and PropSym_subject_level_FIXED498.csv next to it and run:
    python rescale_propsym_to_1to10.py
Outputs (originals untouched): PropSym_image_level_FIXED498_rescaled.csv, PropSym_subject_level_FIXED498_rescaled.csv
"""
import pandas as pd
P=[f"P{i}-Score" for i in range(1,5)]; F=P+[f"S{i}-Score" for i in range(1,7)]
d=pd.read_csv("PropSym_image_level_FIXED498.csv"); s=pd.read_csv("PropSym_subject_level_FIXED498.csv")
d[P]=(1+0.9*d[P]).round(4)
d["Combined-Geometric-Score-1to10"]=d[F].mean(axis=1).round(4)
d["Combined-Geometric-Score-Percent"]=(d["Combined-Geometric-Score-1to10"]*10).round(3)
d["AI-vs-Human-Diff"]=(d["Combined-Geometric-Score-1to10"]-d["Human-Rating-Avg"]).round(4)
g=d.groupby("Subject-ID").agg(n=("Image-Name","count"),pc=("Combined-Geometric-Score-Percent","mean"),a=("Combined-Geometric-Score-1to10","mean"),h=("Human-Rating-Avg","mean"),df=("AI-vs-Human-Diff","mean")).reset_index()
out=s[["Subject-ID"]].merge(g,on="Subject-ID")
out.columns=["Subject-ID","Num-Images","Mean-Combined-Geometric-Score-Percent","Mean-Combined-Geometric-Score-1to10","Mean-Human-Rating","Mean-AI-vs-Human-Diff"]
d.to_csv("PropSym_image_level_FIXED498_rescaled.csv",index=False); out.to_csv("PropSym_subject_level_FIXED498_rescaled.csv",index=False)
print("done")
