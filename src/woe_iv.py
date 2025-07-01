#!/usr/bin/env python
# coding: utf-8

# In[2]:


# src/woe_iv.py

from woe import WOE
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Applies Weight of Evidence (WOE) encoding to specified columns.
    """
    def __init__(self, columns=None):
        self.columns = columns or []
        self.woe = WOE()
        self.fitted = False

    def fit(self, X, y):
        self.woe.fit(X, y, self.columns)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("WOETransformer must be fitted before calling transform.")
        return self.woe.transform(X)


# In[ ]:




