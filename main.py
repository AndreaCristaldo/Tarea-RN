from src.config import Config
from pathlib import Path

def main():
    cfg = Config()
    cfg.create_dirs()

    print("Artifacts debería estar en:")
    print(cfg.artifacts.resolve())

    print("¿Existe artifacts?:", Path(cfg.artifacts).exists())
    print("¿Existe stage1?:", Path(cfg.stage1_dir).exists())
    print("¿Existe stage2?:", Path(cfg.stage2_dir).exists())

if __name__ == "__main__":
    main()
