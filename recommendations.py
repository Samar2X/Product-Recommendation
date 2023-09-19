import numpy as np
import pandas as pd

from typing import List
from models import Dataset

def collaborative_filtering(data:Dataset, th:float=0.5) -> List[List[str]]:
    pred = []

    for i, user in data.test_iteration():
        similar = data.get_similar(i=i, similarity='cosine', th=th)
        
        pred_basket = []
        for basket in data.users.loc[similar, 'basket']:        
            for product in basket:
                if not product in pred_basket:
                    pred_basket.append(product)
        pred.append(pred_basket)
    
    return pred

def content_based_filtering(data:Dataset, th:float=0.3) -> List[List[str]]:
    pred = []

    for i, user in data.test_iteration():
        similar = data.get_similar(i=i, similarity='jaccard', th=th)

        pred_basket = []
        *_, sim_interests = data.users.loc[similar, 'interests']
        *_, sim_basket = data.users.loc[similar, 'basket']
        for interest in sim_interests:
            if interest in user.interests:
                for product in sim_basket:
                    if interest in product:
                        if not product in pred_basket:
                            pred_basket.append(product)
        
        pred.append(pred_basket)
    
    return pred


def popularity_based(data:Dataset) -> List[List[str]]:
    all_products = data.users.loc[data.train.index].basket.sum()
    y_pred = pd.Series(all_products).value_counts().index.tolist()

    return [y_pred for _ in range(len(data.test))]


def svd_reconstruction(matrix:np.array, samples:List) -> np.array:
    U_k, s_k, V_k = np.linalg.svd(matrix, full_matrices=False)
    S_k = np.diag(s_k)
 
    y_pred = U_k.dot(S_k).dot(V_k)
    y_pred[y_pred < 0] = 0
    y_pred[y_pred > 0] = 1
    
    return y_pred[samples]

def svd(data:Dataset) -> List[List[str]]:
    products = sum(list(data.products.products.values()), [])

    sparse = data.sparse.copy()
    sparse.loc[data.test.index] = 0

    sparse = sparse.astype(float).values
    samples = data.test.index.tolist()
    svd_y_pred = svd_reconstruction(matrix=sparse, samples=samples)

    
    svd_y_pred = [list(np.where(r==1)[0]) for r in svd_y_pred]
    svd_y_pred = [[products[i] for i in y_pred] for y_pred in svd_y_pred]

    return svd_y_pred