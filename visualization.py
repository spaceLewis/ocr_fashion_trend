import os

import cv2
import numpy as np
from PIL import Image


# Colormap personnalisé pour 4 classes (valeurs 1 à 4) en BGR
custom_colormap = {
    1: (0, 255, 255),   # Jaune - Hat
    2: (0, 165, 255),   # Orange - Hair
    3: (255, 0, 255),   # Magenta - Sunglasses
    4: (0, 0, 255),     # Rouge - Upper-clothes
    5: (255, 255, 0),   # Cyan - Skirt
    6: (0, 255, 0),     # Vert - Pants
    7: (255, 0, 0),     # Bleu - Dress
    8: (128, 0, 128),   # Violet - Belt
    9: (0, 255, 255),   # Jaune - Left-shoe
    10: (255, 140, 0),  # Orange foncé - Right-shoe
    11: (200, 180, 140), # Beige - Face
    12: (200, 180, 140), # Beige - Left-leg
    13: (200, 180, 140), # Beige - Right-leg
    14: (200, 180, 140), # Beige - Left-arm
    15: (200, 180, 140), # Beige - Right-arm
    16: (0, 128, 255),  # Bleu clair - Bag
    17: (255, 20, 147)  # Rose - Scarf
}

# Légendes associées aux labels
legend_labels = {
    "0": "Background",
    "1": "Hat",
    "2": "Hair",
    "3": "Sunglasses",
    "4": "Upper-clothes",
    "5": "Skirt",
    "6": "Pants",
    "7": "Dress",
    "8": "Belt",
    "9": "Left-shoe",
    "10": "Right-shoe",
    "11": "Face",
    "12": "Left-leg",
    "13": "Right-leg",
    "14": "Left-arm",
    "15": "Right-arm",
    "16": "Bag",
    "17": "Scarf"
}


class SegmentationVisualizer:
    def display_segmented_images_batch(self, original_image_paths, segmentation_masks):
        """
        Affiche les images originales et leurs masques segmentés.

        Args:
            original_image_paths (list): Liste des chemins des images originales.
            segmentation_masks (list): Liste des masques segmentés (NumPy arrays).
        """
        # Matplotlib, ça vous parle ?
        # Alors... au travail ! 😉
        import matplotlib.pyplot as plt
        for img_path, mask in zip(original_image_paths, segmentation_masks):
            if mask is not None:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(Image.open(img_path))
                ax[0].set_title("Image originale")
                ax[0].axis("off")
                ax[1].imshow(mask, cmap="gray")
                ax[1].set_title("Masque segmenté")
                ax[1].axis("off")
                # plt.show() # Affichage si besoin
                # sauvegarde de la comparaison
                comparison_path = f"resultats_segmentation/{os.path.basename(img_path).split('.')[0]}_comparison.png"
                fig.savefig(comparison_path)
                print(f"Comparaison sauvegardée dans {comparison_path}")
            else:
                print(f"Aucun masque disponible pour {img_path}")

    def colorize_mask(self, mask, colormap):
        """
        Applique le colormap personnalisé au masque.
        Pour chaque pixel, s'il correspond à un label défini dans colormap,
        la couleur correspondante est assignée.
        """
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for label, color in colormap.items():
            colored_mask[mask == label] = color
        return colored_mask

    def add_legend(self, image, legend, start_x=10, start_y=10, box_size=15, spacing=5):
        """
        Ajoute une légende sur l'image.
        Pour chaque label, dessine un rectangle de la couleur correspondante et le texte associé.
        """
        img_with_legend = image.copy()
        y = start_y
        for label, text in legend.items():
            # Récupération de la couleur du label
            color = custom_colormap.get(int(label), (255, 255, 255))
            # Dessin d'un petit rectangle rempli
            cv2.rectangle(img_with_legend, (start_x, y), (start_x + box_size, y + box_size), color, -1)
            # Ajout du texte à droite du rectangle
            cv2.putText(img_with_legend, text, (start_x + box_size + spacing, y + box_size - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += box_size + spacing
        return img_with_legend
