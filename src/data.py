"""Descarga, listado, división y pipeline tf.data del dataset TF Flowers."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def download_dataset(cache_dir="data"):
    """Descarga y descomprime el dataset. Devuelve la carpeta con las clases."""
    path = tf.keras.utils.get_file(
        fname="flower_photos", origin=DATASET_URL, untar=True,
        cache_dir=str(Path(cache_dir).resolve()), cache_subdir=".",
    )
    return Path(path)


def list_images(data_dir):
    """Devuelve (rutas, etiquetas, nombres_de_clases)."""
    data_dir = Path(data_dir)
    class_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    paths, labels = [], []
    for idx, name in enumerate(class_names):
        for f in sorted((data_dir / name).glob("*.jpg")):
            paths.append(str(f))
            labels.append(idx)
    return np.array(paths), np.array(labels), class_names


def split_data(paths, labels, val_size=0.15, test_size=0.15, seed=SEED):
    """División estratificada train/val/test (70/15/15 por defecto)."""
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=val_size + test_size, stratify=labels, random_state=seed)
    p_va, p_te, y_va, y_te = train_test_split(
        p_tmp, y_tmp, test_size=test_size / (val_size + test_size), stratify=y_tmp, random_state=seed)
    return {"train": (p_tr, y_tr), "val": (p_va, y_va), "test": (p_te, y_te)}


def make_dataset(paths, labels, img_size=IMG_SIZE, batch_size=BATCH_SIZE, training=False):
    """Crea un tf.data.Dataset (imágenes float32 en [0, 255]; la normalización va en el modelo)."""
    def _load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, img_size)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    return ds.map(_load, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)


def build_datasets(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    paths, labels, class_names = list_images(data_dir)
    splits = split_data(paths, labels)
    datasets = {
        name: make_dataset(p, y, img_size, batch_size, training=(name == "train"))
        for name, (p, y) in splits.items()
    }
    return datasets, splits, class_names
