from sklearn.preprocessing import StandardScaler

def scaler_data(X_train,y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_scaled = scaler.fit_transform(y_train)
    return X_train_scaled,y_train_scaled