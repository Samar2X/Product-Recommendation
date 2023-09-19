import random
import numpy as np
import pandas as pd

from typing import List, Tuple, Iterator
from similarities import cosine_similarity, pairwise_jaccard

INTERESTS = ['Home decor', 'Technology', 'Cars', 'Musical instrument', 'Sports']

class Products():
    def __init__(self, n_products:int=20) -> None:
        self.prod_classes = int(n_products / len(INTERESTS))
        self.products = {
            interest:
                [f'{interest} product {i+1}' for i in range(self.prod_classes)]
                    for interest in INTERESTS
            }
        
    def get_random_products(self, prod_class:str):
        n_items = random.choice([i for i in range(1, 2)])
        samples = random.sample(self.products[prod_class], k=n_items)

        return samples

class User():
    def __init__(self, products) -> None:
        self.age = random.randint(16, 70)
        self.gender = random.choice(['male', 'female'])
        self.interests = self._get_random_interests()
        self.products = products
        self.basket = self._get_random_basket()

    def _get_random_interests(self, n_interests:int=3) -> List[str]:
        n_interests = random.choice([i+1 for i in range(n_interests)])  # 1, 2, 3
        return random.sample(INTERESTS, k=n_interests)

    def _get_random_basket(self) -> List[str]:
        
        '''
            Correspond to interactions
            A user has a random number of products
            Products have a unique ascending id in range {n_products}
        '''
        basket = []
        for interest in self.interests:
            basket.extend(self.products.get_random_products(interest))

        return sorted(basket)

    def to_list(self) -> Tuple[int, str, List[str]]:
        return self.age, self.gender, self.interests, self.basket


class Dataset():
    def __init__(self, n_users:int=1000, n_products:int=20) -> None:
        data = []
        self.products = Products(n_products=n_products)
        for _ in range(n_users):
            data.append(User(self.products))
        data = [user.to_list() for user in data]

        cols = ['age', 'gender', 'interests', 'basket']
        self.users = pd.DataFrame(data=data, columns=cols)
        self.n_products = pd.Series(self.users.basket.sum()).unique().size

        self.sparse = self._get_sparse_array()        
        self.train, self.test = self._create_splits()
        self.cos_sim = self._calculate_cosine_similarities()
        self.jac_sim = self._calculate_jaccard_similarities()
    
    
    def to_one_hot(self, pred:List[List[int]]) -> np.array:
        y_pred = np.zeros((len(pred), self.n_products))
        for i, p in enumerate(pred):
            y_pred[i, p] = 1
        
        return y_pred
            
    def get_similar(self, i:int, similarity:str, th:float=0.5) -> np.array:
        sim = self.cos_sim if similarity=='cosine' else self.jac_sim
        similar = (-sim[i]).argsort()

        # only return similar users based on the threshold
        if th:
            similar = similar[sim[i,similar] > th]

        # only train instances
        _, *similar = np.intersect1d(similar, self.train.index)
        return similar
    

    def test_iteration(self) -> Iterator:
        data = self.users.loc[self.test.index.sort_values()]
        return iter(data.iterrows())
    
    
    def _get_sparse_array(self) -> pd.DataFrame:
        dummies = pd.Series(self.users.basket.explode(), dtype=int)
        sparse = pd.get_dummies(dummies).groupby(dummies.index).agg(sum)
        
        return sparse
    

    def _create_splits(self, test_pct:float=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test = self.sparse.sample(frac=test_pct)        
        train = self.sparse.loc[~self.sparse.index.isin(test.index)]

        return train, test
    

    def _calculate_cosine_similarities(self) -> np.array:
        sparse = self.sparse.values
        return cosine_similarity(sparse)
    

    def _calculate_jaccard_similarities(self) -> np.array:
        jac_sim = self.users.interests.apply(pairwise_jaccard, interests=self.users.interests.tolist())
        return np.stack(jac_sim)
    