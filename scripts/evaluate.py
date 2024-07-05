import sys
import pandas as pd
import numpy as np

from evaluation import runEvaluate

ref_path = '/export/livia/home/vision/Ymohammadi/Dataset/sketch.npz'

data_list = []

for iteration in np.arange(0, 501, 25):
    sample_path = f'/export/livia/home/vision/Ymohammadi/Code/results/samples/samples_{iteration}.npz'
    results = runEvaluate(ref_path, sample_path, verbose=True)
    data_list.append(results)

df = pd.DataFrame(data_list)
csv_file = f"/export/livia/home/vision/Ymohammadi/Code/results/evaluation.csv"
df.to_csv(csv_file, index=False)

print(f"written to {csv_file}")