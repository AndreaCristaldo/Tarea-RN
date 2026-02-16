import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks

from src.config import Config
from src.data import load_fashion_mnist_flat
from src.models import build_classifier
from src.utils import set_seed, ensure_dir, save_json

CLASS_NAMES = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

def main():
    cfg = Config()
    set_seed(cfg.seed)
    ensure_dir(cfg.stage2_dir)

    x_train, y_train, x_test, y_test = load_fashion_mnist_flat()

    # Cargar encoder entrenado en etapa 1
    encoder = tf.keras.models.load_model(cfg.stage1_dir / "encoder.keras")
    encoder.trainable = False  # <- requisito: congelado

    classifier = build_classifier(encoder, cfg.num_classes, cfg.dropout)

    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.clf_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cbs = [
        callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor="val_accuracy"),
        callbacks.ModelCheckpoint(str(cfg.stage2_dir / "classifier.keras"), save_best_only=True, monitor="val_accuracy"),
        callbacks.CSVLogger(str(cfg.stage2_dir / "history.csv"))
    ]

    history = classifier.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=cfg.clf_epochs,
        batch_size=cfg.clf_batch_size,
        shuffle=True,
        callbacks=cbs,
        verbose=1
    )

    test_loss, test_acc = classifier.evaluate(x_test, y_test, verbose=0)

    save_json(cfg.stage2_dir / "metrics.json", {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "encoder_frozen": True
    })

    # Curva accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(cfg.stage2_dir / "acc_curve.png", dpi=150)
    plt.close()

    # Predicciones visibles (10 ejemplos)
    preds = classifier.predict(x_test[:10], verbose=0).argmax(axis=1)
    with open(cfg.stage2_dir / "pred_samples.txt", "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(f"{i}: pred={CLASS_NAMES[preds[i]]} true={CLASS_NAMES[int(y_test[i])]}\n")

    print("\n✅ ETAPA 2 OK")
    print(f"Test accuracy: {test_acc:.4f}")
    print("Abrí estos archivos para ver resultados:")
    print(" -", cfg.stage2_dir / "acc_curve.png")
    print(" -", cfg.stage2_dir / "pred_samples.txt")
    print("Modelo:", cfg.stage2_dir / "classifier.keras")

if __name__ == "__main__":
    main()
