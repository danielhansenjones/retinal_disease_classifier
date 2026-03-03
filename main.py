from src.config import Config
from src.train import train

if __name__ == "__main__":
    config = Config()
    train(config)