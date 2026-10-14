"""Entrenamiento completo: python -m src.train --epochs 30"""
import argparse
import json
from pathlib import Path

import tensorflow as tf

from src import data, evaluate, model as model_lib

ROOT = Path(__file__).resolve().parents[1]


def main(epochs, batch_size, img_size):
    tf.keras.utils.set_random_seed(data.SEED)
    size = (img_size, img_size)

    data_dir = data.download_dataset(ROOT / "data")
    ds, splits, class_names = data.build_datasets(data_dir, size, batch_size)
    print({k: len(v[0]) for k, v in splits.items()}, class_names)

    model = model_lib.build_cnn(size + (3,), len(class_names))
    model.summary()

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    ckpt = ROOT / "models" / "cnn_flores.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    history = model.fit(ds["train"], validation_data=ds["val"], epochs=epochs, callbacks=callbacks)

    model.save(ckpt)  # pesos restaurados de la mejor época
    (ROOT / "models" / "class_names.json").write_text(json.dumps(class_names), encoding="utf-8")
    (ROOT / "reports" / "history.json").write_text(json.dumps(history.history), encoding="utf-8")

    evaluate.plot_history(history, ROOT / "reports" / "training_curves.png")
    y_true, y_pred, _ = evaluate.get_predictions(model, ds["test"])
    evaluate.plot_confusion_matrix(y_true, y_pred, class_names, ROOT / "reports" / "confusion_matrix.png")
    report = evaluate.text_report(y_true, y_pred, class_names)
    (ROOT / "reports" / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print("Test:", dict(zip(model.metrics_names, model.evaluate(ds["test"], verbose=0))))
    p_te, y_te = splits["test"]
    evaluate.show_predictions(model, p_te, y_te, class_names, n=12, img_size=size,
                              path=ROOT / "reports" / "predictions.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=180)
    a = ap.parse_args()
    main(a.epochs, a.batch_size, a.img_size)
