import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from typing import List

def pairwise_jaccard(a:List[str], interests:pd.Series):
    return np.asarray([jaccard_similarity(a,b) for b in interests])

def jaccard_similarity(a:List[str], b:List[str]) -> float:
    a = set(a)
    b = set(b)
    return float(len(a.intersection(b)) / len(a.union(b)))