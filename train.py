from sklearn.model_selection import train_test_split
from dataset import load_dataset
from preprocess import preprocess_data
from model import build_model
from utils import save_model
import config

def train():
    X, y = load_dataset()
    X = preprocess_data(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    model = build_model()
    model.fit(X_train, y_train)

    save_model(model, config.MODEL_PATH)
    print("✅ Model trained and saved")

if __name__ == "__main__":
    train()
