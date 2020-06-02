from sklearn.base import TransformerMixin


def dict_get(obj, key, default):
    if obj is None:
        return default
    if key in obj:
        toret = obj[key]
        if toret is None:
            return default
        return toret
    else:
        return default
    
class MultiLabelEncoder(TransformerMixin):
    def __init__(self, inplace=False):
        self.inplace = inplace

    def fit(self, X, y=None):
        self.encoder = {}
        self.cols = [c for c in X.columns if X[c].dtype.name == 'object']
        for col in self.cols:
            col_enc = {}
            count = 1
            unique = list(X[col].unique())
            for u in unique:
                col_enc[u] = count
                count += 1
            self.encoder[col] = col_enc

        return self

    def transform(self, X):
        if self.inplace:
            temp = X
        else:
            temp = X.copy()

        for col in self.cols:
            temp[col] = temp[col].apply(lambda x: dict_get(self.encoder[col], x, 0))

        return temp
    
class ColumnsSelector(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.feature_names = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.cols]