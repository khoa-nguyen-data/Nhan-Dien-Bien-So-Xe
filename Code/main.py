from preprocess import preprocess_data
from train_model import train
from evaluate import evaluate

def main():
    X, y = preprocess_data()
    model = train(X, y)
    evaluate(model, X, y)

if __name__ == "__main__":
    main()
