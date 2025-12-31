from sklearn.preprocessing import StandardScaler

def preprocess_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)
