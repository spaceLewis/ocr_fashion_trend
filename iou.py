"""Compare deux masques PNG contenant les identifiants de classe de main.py.

Exemple : python iou.py assets\\Mask\\mask_image_0.png assets\\Reference\\mask_image_0.png
Le second masque doit être une annotation de référence indépendante du modèle.
Les pixels contiennent les identifiants 0 à 17 ; 0 correspond au fond.
Les images RGB ou RGBA opaques sont acceptées si leurs canaux RGB sont identiques.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path):
    """Lit les identifiants de classe sans convertir les couleurs ambiguës."""
    with Image.open(path) as image:
        mask = np.array(image)
        if image.mode in ("RGB", "RGBA"):
            rgb = mask[:, :, :3]
            if not np.all(rgb == rgb[:, :, :1]):
                raise ValueError(f"Le masque {path} contient des couleurs ambiguës : les canaux RGB doivent être identiques.")
            if image.mode == "RGBA" and not np.all(mask[:, :, 3] == 255):
                raise ValueError(f"Le masque {path} contient des pixels transparents.")
            mask = rgb[:, :, 0]
    return mask


def calculate_iou(prediction, reference):
    """Retourne l'IoU des classes présentes dans au moins un des deux masques."""
    if prediction.ndim != 2 or reference.ndim != 2:
        raise ValueError("Les masques doivent contenir un identifiant par pixel.")
    if prediction.shape != reference.shape:
        raise ValueError("Les masques doivent avoir les mêmes dimensions.")
    if not np.all(np.isin(prediction, range(18))) or not np.all(np.isin(reference, range(18))):
        raise ValueError("Les identifiants de classe doivent être compris entre 0 et 17.")

    scores = {}
    for class_id in range(18):
        predicted_class = prediction == class_id
        reference_class = reference == class_id
        union = np.count_nonzero(predicted_class | reference_class)
        if union:
            intersection = np.count_nonzero(predicted_class & reference_class)
            scores[class_id] = intersection / union
    return scores


def compare_directories(prediction_dir, reference_dir):
    """Compare tous les masques appariés par nom dans deux dossiers."""
    prediction_dir = Path(prediction_dir)
    reference_dir = Path(reference_dir)
    if not prediction_dir.is_dir() or not reference_dir.is_dir():
        raise ValueError("Les dossiers de masques prédits et de référence doivent exister.")

    prediction_files = {path.name: path for path in prediction_dir.iterdir() if path.is_file()}
    reference_files = {path.name: path for path in reference_dir.iterdir() if path.is_file()}
    missing_references = sorted(prediction_files.keys() - reference_files.keys())
    missing_predictions = sorted(reference_files.keys() - prediction_files.keys())
    if missing_references or missing_predictions:
        raise ValueError(
            "Les fichiers doivent être appariés par nom. "
            f"Références manquantes : {missing_references}; "
            f"prédictions manquantes : {missing_predictions}."
        )
    if not prediction_files:
        raise ValueError("Aucun fichier à comparer dans les dossiers.")

    results = {}
    for filename in sorted(prediction_files):
        prediction = load_mask(prediction_files[filename])
        reference = load_mask(reference_files[filename])
        results[filename] = calculate_iou(prediction, reference)
    return results


def main():
    parser = argparse.ArgumentParser(description="Calculer l'IoU entre deux masques de segmentation.")
    parser.add_argument("prediction", help="Masque prédit en PNG")
    parser.add_argument("reference", help="Masque de référence annoté en PNG")
    args = parser.parse_args()

    prediction = load_mask(args.prediction)
    reference = load_mask(args.reference)

    scores = calculate_iou(prediction, reference)
    for class_id, score in scores.items():
        print(f"Classe {class_id} : IoU = {score:.1%}")

    foreground_scores = [score for class_id, score in scores.items() if class_id != 0]
    if foreground_scores:
        print(f"IoU moyenne hors fond : {np.mean(foreground_scores):.1%}")
    else:
        print("IoU moyenne hors fond indisponible : aucune classe de vêtement ou de personne présente.")


if __name__ == "__main__":
    main()
