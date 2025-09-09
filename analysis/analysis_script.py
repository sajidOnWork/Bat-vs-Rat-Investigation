import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

D1_PATH = os.path.join(DATA_DIR, 'dataset1.csv')
D2_PATH = os.path.join(DATA_DIR, 'dataset2.csv')

print('Loading datasets...')
d1 = pd.read_csv(D1_PATH)
d2 = pd.read_csv(D2_PATH)
print('dataset1 rows:', len(d1))
print('dataset2 rows:', len(d2))
