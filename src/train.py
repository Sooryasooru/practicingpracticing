from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from src.preprocess import scale_data

def train_model():
    X,y = load_iris(return_X_y = True)
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size = 0.2)
    X_train,X_test = scale_data(X_train,X_test)
    model = LogisticRegression(max_iter = 200)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)

    return acc

if __name__ == "__main__":
    print("accuracy",train_model())