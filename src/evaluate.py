"""Métricas, gráficas, matriz de confusión y visualización de predicciones."""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


def plot_history(history, path=None):
    h = history.history if hasattr(history, "history") else history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h["accuracy"], label="Entrenamiento")
    ax[0].plot(h["val_accuracy"], label="Validación")
    ax[0].set(title="Accuracy", xlabel="Época", ylabel="Accuracy")
    ax[0].legend()
    ax[1].plot(h["loss"], label="Entrenamiento")
    ax[1].plot(h["val_loss"], label="Validación")
    ax[1].set(title="Pérdida (loss)", xlabel="Época", ylabel="Loss")
    ax[1].legend()
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def get_predictions(model, ds):
    """Devuelve (y_true, y_pred, probabilidades) sobre un dataset sin barajar."""
    y_true = np.concatenate([y.numpy() for _, y in ds])
    probs = model.predict(ds, verbose=0)
    return y_true, probs.argmax(axis=1), probs


def plot_confusion_matrix(y_true, y_pred, class_names, path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title("Matriz de confusión (conjunto de prueba)")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def text_report(y_true, y_pred, class_names):
    return classification_report(y_true, y_pred, target_names=class_names, digits=3)


def show_predictions(model, paths, labels, class_names, n=12, img_size=(180, 180), path=None, seed=0):
    """Muestra n predicciones sobre imágenes aleatorias (verde = acierto, rojo = error)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=n, replace=False)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    for ax, i in zip(np.ravel(axes), idx):
        img = Image.open(paths[i]).convert("RGB")
        x = np.expand_dims(np.asarray(img.resize(img_size), dtype="float32"), 0)
        probs = model.predict(x, verbose=0)[0]
        pred = int(probs.argmax())
        ok = pred == int(labels[i])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Real: {class_names[labels[i]]}\nPred: {class_names[pred]} ({probs[pred]:.0%})",
                     color="green" if ok else "red", fontsize=9)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
