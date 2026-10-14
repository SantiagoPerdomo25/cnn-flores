"""Carga del modelo y predicción sobre una imagen (módulo y CLI)."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "cnn_flores.keras"
CLASSES_PATH = ROOT / "models" / "class_names.json"


def load_artifacts(model_path=MODEL_PATH, classes_path=CLASSES_PATH):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    class_names = json.loads(Path(classes_path).read_text(encoding="utf-8"))
    return model, class_names


def predict_image(model, image, class_names):
    """image: PIL.Image. Devuelve clase, confianza y todas las probabilidades."""
    h, w = model.input_shape[1:3]
    img = image.convert("RGB").resize((w, h))
    x = np.expand_dims(np.asarray(img, dtype="float32"), 0)  # 0-255, el modelo normaliza
    probs = model.predict(x, verbose=0)[0]
    top = int(probs.argmax())
    return {
        "class": class_names[top],
        "confidence": float(probs[top]),
        "probabilities": {c: float(p) for c, p in zip(class_names, probs)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predice la clase de una imagen.")
    parser.add_argument("image", help="Ruta de la imagen")
    args = parser.parse_args()
    model, names = load_artifacts()
    res = predict_image(model, Image.open(args.image), names)
    print(f"Predicción: {res['class']} ({res['confidence']:.2%})")
    for c, p in sorted(res["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:<12} {p:.2%}")
