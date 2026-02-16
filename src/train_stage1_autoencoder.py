import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks

from src.config import Config
from src.data import load_fashion_mnist_flat
from src.models import build_autoencoder
from src.utils import set_seed, ensure_dir, save_json

def main():
    cfg = Config()
    set_seed(cfg.seed)
    ensure_dir(cfg.stage1_dir)

    x_train, _, x_test, _ = load_fashion_mnist_flat()

    autoencoder, encoder, decoder = build_autoencoder(cfg.input_dim, cfg.latent_dim)

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.ae_lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse")]
    )

    cbs = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
        callbacks.ModelCheckpoint(str(cfg.stage1_dir / "autoencoder.keras"), save_best_only=True, monitor="val_loss"),
        callbacks.CSVLogger(str(cfg.stage1_dir / "history.csv"))
    ]

    history = autoencoder.fit(
        x_train, x_train,
        validation_data=(x_test, x_test),
        epochs=cfg.ae_epochs,
        batch_size=cfg.ae_batch_size,
        shuffle=True,
        callbacks=cbs,
        verbose=1
    )

    # Guardar encoder/decoder para transfer learning
    encoder.save(cfg.stage1_dir / "encoder.keras")
    decoder.save(cfg.stage1_dir / "decoder.keras")

    # Guardar métricas
    save_json(cfg.stage1_dir / "metrics.json", {
        "best_val_loss": float(min(history.history["val_loss"])),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "latent_dim": cfg.latent_dim
    })

    # Curva loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(cfg.stage1_dir / "loss_curve.png", dpi=150)
    plt.close()

    # Reconstrucciones visibles
    idx = np.random.choice(len(x_test), size=8, replace=False)
    x = x_test[idx]
    recon = autoencoder.predict(x, verbose=0)

    plt.figure(figsize=(8, 2))
    for i in range(8):
        ax = plt.subplot(2, 8, i+1)
        ax.imshow(x[i].reshape(28,28), cmap="gray")
        ax.axis("off")
        ax = plt.subplot(2, 8, 8+i+1)
        ax.imshow(recon[i].reshape(28,28), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(cfg.stage1_dir / "recon_samples.png", dpi=150)
    plt.close()

    print("\n✅ ETAPA 1 OK")
    print("Abrí estos archivos para ver resultados:")
    print(" -", cfg.stage1_dir / "loss_curve.png")
    print(" -", cfg.stage1_dir / "recon_samples.png")
    print("Modelos:", cfg.stage1_dir / "encoder.keras", "y", cfg.stage1_dir / "autoencoder.keras")

if __name__ == "__main__":
    main()