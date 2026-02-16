
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import Config
from src.data import load_fashion_mnist_flat

CLASS_NAMES = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

def main():
    cfg = Config()

    # 1) Verificar que existe el modelo final
    model_path = cfg.stage2_dir / "classifier.keras"
    if not model_path.exists():
        print("❌ No existe el modelo final:", model_path)
        print("Ejecutá primero:")
        print("  python -m src.train_stage1_autoencoder")
        print("  python -m src.train_stage2_classifier")
        return

    # 2) Cargar datos de test
    _, _, x_test, y_test = load_fashion_mnist_flat()

    # 3) Cargar modelo y evaluar
    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Test loss={loss:.4f} | Test accuracy={acc:.4f}")

    # 4) Mostrar predicciones de ejemplo en consola
    n = 12
    probs = model.predict(x_test[:n], verbose=0)
    preds = probs.argmax(axis=1)

    print("\nEjemplos (pred vs true):")
    for i in range(n):
        print(f"{i:2d}: pred={CLASS_NAMES[int(preds[i])]:12s} | true={CLASS_NAMES[int(y_test[i])]:12s}")

    # 5) Guardar una grilla (visual) para evidencia
    out_img = cfg.stage2_dir / "pred_grid.png"

    # Para mostrar imágenes hay que re-formar a 28x28
    imgs = x_test[:n].reshape(-1, 28, 28)

    cols = 6
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(12, 4))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(imgs[i], cmap="gray")
        ax.set_title(f"P:{preds[i]} T:{y_test[i]}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    plt.close()

    print("\n🖼️ Grilla guardada en:", out_img)
    print("Abrila en VS Code para ver predicciones visuales.")

if __name__ == "__main__":
    main()
