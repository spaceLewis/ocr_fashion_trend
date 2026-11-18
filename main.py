
import os
import time

import numpy as np
from dotenv import load_dotenv

import cv2
from iou import compare_directories
from segmentation import ImageSegmenter
from visualization import SegmentationVisualizer, custom_colormap, legend_labels

start_time = time.perf_counter()
COST_PER_SECOND = 0.00012

load_dotenv()
only_test = os.getenv("only_test", "false").strip().lower() == "true"

# TODO: Modifiez ces valeurs selon votre configuration
image_dir = "assets/IMG"  # Exemple : si vous êtes sur Colab et avez uploadé un dossier
max_images = 50 # nombre maximum d'images

api_token = os.getenv("token")

# Créons le dossier d'images s'il n'existe pas (pour l'exemple)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
    print(f"Dossier '{image_dir}' créé. Veuillez y ajouter des images .jpg ou .png.")
else:
    print(f"Dossier '{image_dir}' existant.")

if not api_token:
    print("\nATTENTION : Ajoutez la variable 'token' contenant votre token API dans le fichier .env.")

##########################################################################################################

# API_URL = "https://api-inference.huggingface.co/models/sayeed99/segformer_b3_clothes" # Remplacez ... par le bon endpoint.
API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"
API_URL_LOCAL = "http://localhost:8000/segment"  # Renvoie le JSON (label/mask) comme l'API Hugging Face

# Mode local si api_local=true dans .env, sinon mode online (Hugging Face)
api_local = os.getenv("api_local", "false").strip().lower() == "true"
api_url = API_URL
if api_local:
    api_url = API_URL_LOCAL
print(f"Mode : {'local' if api_local else 'online'} -> {api_url}")

headers = {
    "Authorization": f"Bearer {api_token}"
    # Le "Content-Type" sera ajouté dynamiquement lors de l'envoi de l'image
}
headers["Authorization"] = f"Bearer {api_token or ''}"
image_segmenter = ImageSegmenter(api_local, api_url, headers, COST_PER_SECOND)
visualizer = SegmentationVisualizer()

# Lister les chemins des images à traiter
# Assurez-vous d'avoir des images dans le dossier 'image_dir'!
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))][:max_images] # A vous de jouer !

if not image_paths:
    print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
else:
    print(f"{len(image_paths)} image(s) à traiter : {image_paths}")

##########################################################################################################

print(f"Valeur de only_test : {only_test}")
if only_test == True:
    print("\nMode test uniquement activé.")
    print("Seules les évaluations IoU seront effectuées.")
else:
    print("\nMode test uniquement désactivé.")
    print("Toutes les étapes de traitement seront effectuées.")

    # Appeler la fonction pour segmenter les images listées dans image_paths
    if image_paths:
        print(f"\nTraitement de {len(image_paths)} image(s) en batch...")
        batch_seg_results = image_segmenter.segment_images_batch(image_paths)
        print("Traitement en batch terminé.")
    else:
        batch_seg_results = []
        print("Aucune image à traiter en batch.")

    # Afficher les résultats du batch
    print("Affichage des résultats du batch...")
    if batch_seg_results:
        visualizer.display_segmented_images_batch(image_paths, batch_seg_results)
    else:
        print("Aucun résultat de segmentation à afficher.")

    # Lecture des paires image/mask depuis les dossiers
    image_files = sorted(os.listdir("assets/IMG"))[:max_images]
    mask_files = sorted(os.listdir("assets/Mask"))[:max_images]

    paires = []
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join("assets/IMG", img_file)
        mask_path = os.path.join("assets/Mask", mask_file)

        image = cv2.imread(img_path)                       # Image en couleur
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)   # Masque en niveaux de gris

        paires.append((image, mask))

    print("Lecture des paires image/mask effectuée.")

    # Coloriser les masques et ajouter la légende
    # Application du colormap, ajout de la légende et superposition image/mask

    for idx, (img, msk) in enumerate(paires):
        # Colorisation du masque avec le colormap personnalisé
        colored_mask = visualizer.colorize_mask(msk, custom_colormap)

        # Ajout de la légende sur le masque colorisé
        colored_mask_with_legend = visualizer.add_legend(colored_mask, legend_labels)

        # Superposition du masque coloré sur l'image originale
        overlay = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)
        overlay_with_legend = visualizer.add_legend(overlay, legend_labels)

        # Concatenation des images sur une seule ligne
        concatenated = np.hstack([img, colored_mask_with_legend, overlay_with_legend])

        # Sauvegarde des résultats dans resultats_segmentation
        print(f"Résultat pour la paire {idx} :")
        result_path = os.path.join("resultats_segmentation", f"result_{idx}.png")
        cv2.imwrite(result_path, concatenated)
        print(f"Résultat sauvegardé dans {result_path}")

    print("Tous les résultats ont été sauvegardés.")
    total_elapsed = time.perf_counter() - start_time
    print(f"Temps total d'exécution : {total_elapsed:.2f} secondes.")
    print(f"Coût total estimé : {total_elapsed * COST_PER_SECOND:.6f} $.")

##########################################################################################################

# test IoU (Intersection over Union) pour évaluer la qualité de la segmentation
print("\nÉvaluation IoU des masques prédits par rapport aux références...")
iou_results = compare_directories("assets/Mask", "assets/Reference")
all_foreground_scores = []
class_scores = {}

for filename, scores in iou_results.items():
    foreground_scores = [score for class_id, score in scores.items() if class_id != 0]
    if foreground_scores:
        image_mean = np.mean(foreground_scores)
        all_foreground_scores.extend(foreground_scores)
        print(f"{filename} : IoU moyenne hors fond = {image_mean:.1%}")
    else:
        print(f"{filename} : IoU moyenne hors fond indisponible (aucune classe présente).")

    for class_id, score in scores.items():
        if class_id != 0:
            class_scores.setdefault(class_id, []).append(score)

print(f"\nBilan IoU : {len(iou_results)} paire(s) comparée(s).")
if all_foreground_scores:
    print(f"IoU moyenne globale hors fond : {np.mean(all_foreground_scores):.1%}")
    for class_id, scores in sorted(class_scores.items()):
        class_name = legend_labels[str(class_id)]
        print(f"  Classe {class_id} ({class_name}) : IoU moyenne = {np.mean(scores):.1%}")
else:
    print("IoU moyenne globale hors fond indisponible : aucune classe présente.")
