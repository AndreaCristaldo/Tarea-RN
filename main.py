from src.config import Config

def main():
    cfg = Config()
    cfg.create_dirs()
    print("✔ Directorios de artifacts creados en:", cfg.artifacts.resolve())

if __name__ == "__main__":
    main()
