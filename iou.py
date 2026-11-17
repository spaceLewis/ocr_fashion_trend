"""Compare deux masques PNG contenant les identifiants de classe de main.py.

Exemple : python iou.py assets\\Mask\\mask_image_0.png assets\\Reference\\mask_image_0.png
Le second masque doit être une annotation de référence indépendante du modèle.
Les pixels contiennent les identifiants 0 à 17 ; 0 correspond au fond.
"""

import argparse

import numpy as np
from PIL import Image


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


def main():
    parser = argparse.ArgumentParser(description="Calculer l'IoU entre deux masques de segmentation.")
    parser.add_argument("prediction", help="Masque prédit en PNG")
    parser.add_argument("reference", help="Masque de référence annoté en PNG")
    args = parser.parse_args()

    with Image.open(args.prediction) as image:
        prediction = np.array(image)
    with Image.open(args.reference) as image:
        reference = np.array(image)

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
