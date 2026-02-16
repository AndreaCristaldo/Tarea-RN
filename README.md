# Transfer Learning con Autoencoder dataset Fashion-MNIST

## Objetivo
Aplicar transfer learning en 2 etapas:
1) Entrenar un Autoencoder para extracción de características.
2) Congelar encoder + vector latente y entrenar unas capas más con salida de clasificación.


### Etapa 1
python -m src.train_stage1_autoencoder

outputs:
- artifacts/stage1/encoder.keras
- artifacts/stage1/decoder.keras
- artifacts/stage1/autoencoder.keras

### Etapa 2
python -m src.train_stage2_classifier

outputs:
- artifacts/stage2/classifier.keras

### Evaluación
python -m src.evaluate
