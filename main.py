
import base64
import io
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

import cv2
from iou import compare_directories

start_time = time.perf_counter()
COST_PER_SECOND = 0.00012

load_dotenv()
only_test = os.getenv("only_test", "false").strip().lower() == "true"

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

# TODO: Modifiez ces valeurs selon votre configuration
image_dir = "assets/IMG"  # Exemple : si vous êtes sur Colab et avez uploadé un dossier
max_images = 50  # nombre maximum d'images

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

# Lister les chemins des images à traiter
# Assurez-vous d'avoir des images dans le dossier 'image_dir'!
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))][:max_images] # A vous de jouer !

if not image_paths:
    print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
else:
    print(f"{len(image_paths)} image(s) à traiter : {image_paths}")

##########################################################################################################

CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17
}

def get_image_dimensions(img_path):
    """
    Get the dimensions of an image.

    Args:
        img_path (str): Path to the image.

    Returns:
        tuple: (width, height) of the image.
    """
    original_image = Image.open(img_path)
    return original_image.size

def decode_base64_mask(base64_string, width, height):
    """
    Decode a base64-encoded mask into a NumPy array.

    Args:
        base64_string (str): Base64-encoded mask.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Single-channel mask array.
    """
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)

def create_masks(results, width, height):
    """
    Combine multiple class masks into a single segmentation mask.

    Args:
        results (list): List of dictionaries with 'label' and 'mask' keys.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Combined segmentation mask with class indices.
    """
    combined_mask = np.zeros((height, width), dtype=np.uint8)  # Initialize with Background (0)

    # Process non-Background masks first
    for result in results:
        label = result['label']
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:  # Skip Background
            continue
        mask_array = decode_base64_mask(result['mask'], width, height)
        combined_mask[mask_array > 0] = class_id

    # Process Background last to ensure it doesn't overwrite other classes unnecessarily
    # (Though the model usually provides non-overlapping masks for distinct classes other than background)
    for result in results:
        if result['label'] == 'Background':
            mask_array = decode_base64_mask(result['mask'], width, height)
            # Apply background only where no other class has been assigned yet
            # This logic might need adjustment based on how the model defines 'Background'
            # For this model, it seems safer to just let non-background overwrite it first.
            # A simple application like this should be fine: if Background mask says pixel is BG, set it to 0.
            # However, a more robust way might be to only set to background if combined_mask is still 0 (initial value)
            combined_mask[mask_array > 0] = 0 # Class ID for Background is 0

    return combined_mask

##########################################################################################################

def process_single_image(image_paths, api_local, api_url, headers):

    if image_paths:
        # single_image_path = image_paths[0] # première image de la liste
        # single_image_path = image_paths[0] # première image de la liste
        single_image_path = image_paths
        print(f"Traitement de l'image : {single_image_path}")

        try:
            with open(single_image_path, "rb") as image_file:  # Ouvrir l'image en mode binaire pour la lecture
                image_data = image_file.read()

            # Déterminer le `Content-Type` (jpeg ou png).
            if single_image_path.lower().endswith(".png"):
                content_type = "image/png"
            elif single_image_path.lower().endswith(".jpg") or single_image_path.lower().endswith(".jpeg"):
                content_type = "image/jpeg"
            else:
                raise ValueError(f"Unsupported image format for file: {single_image_path}")

            if api_local:
                # L'API locale (FastAPI) attend un upload multipart avec le champ "file"
                response = requests.post(
                    api_url,
                    files={"file": (os.path.basename(single_image_path), image_data, content_type)},
                    timeout=120,
                )
            else:
                response = requests.post(
                    api_url,
                    headers={**headers, "Content-Type": content_type},
                    data=image_data,
                    timeout=120,
                )
            response.raise_for_status()

            #  Vérification du code de statut de la réponse
            print(f"Code de statut de la réponse : {response.status_code}")
            if response.status_code != 200:
                raise ValueError(f"Erreur lors de l'appel à l'API Hugging Face : {response.status_code} - {response.text}")

            segmentation_results = response.json()
            if not isinstance(segmentation_results, list):  # Vérifie que la réponse est bien une liste
                raise ValueError(f"Réponse inattendue de l'API Hugging Face : {segmentation_results}")

            # création du masque à partir des résultats de segmentation
            width, height = get_image_dimensions(single_image_path)
            segmentation_mask = create_masks(segmentation_results, width, height)
            print("Masque généré avec succès.")
            # en option, sauvegarder le masque en image
            segmentation_mask_image = Image.fromarray(segmentation_mask.astype(np.uint8)) # Convertir le masque en image
            # sauvegarde du masque avec un nom basé sur le nom de l'image originale
            segmentation_mask_image.save(f"assets/Mask/mask_{os.path.basename(single_image_path)}")
            print(f"Masque sauvegardé sous 'assets/Mask/mask_{os.path.basename(single_image_path)}'.")
            return segmentation_mask

        except Exception as e:
            print(f"Une erreur est survenue : {e}")
    else:
        print("Aucune image à traiter. Vérifiez la configuration de 'image_dir' et 'max_images'.")

##########################################################################################################

def segment_images_batch(list_of_image_paths):
    """
    Segmente une liste d'images en utilisant l'API Hugging Face.

    Args:
        list_of_image_paths (list): Liste des chemins vers les images.

    Returns:
        list: Liste des masques de segmentation (tableaux NumPy).
              Contient None si une image n'a pas pu être traitée.
    """
    batch_segmentations = []
    successful_images = 0
    successful_elapsed = 0.0

    # N'oubliez pas de mettre une pause entre chaque appel API !
    for single_image_path in list_of_image_paths:
        image_start_time = time.perf_counter()
        segmentation_mask = None
        try:
            segmentation_mask = process_single_image(single_image_path, api_local, api_url, headers)
            batch_segmentations.append(segmentation_mask)
        except Exception as e:
            print(f"Erreur lors du traitement de {single_image_path} : {e}")
            batch_segmentations.append(None)
        finally:
            elapsed = time.perf_counter() - image_start_time
            if segmentation_mask is not None:
                successful_images += 1
                successful_elapsed += elapsed
            print(f"Temps de traitement de {single_image_path} : {elapsed:.2f} secondes.")
            print(f"Coût estimé pour {single_image_path} : {elapsed * COST_PER_SECOND:.6f} $.")
    if successful_images:
        estimated_cost = successful_elapsed / successful_images * 500_000 * COST_PER_SECOND
        print(f"Coût estimé pour 500 000 images (moyenne des segmentations réussies) : {estimated_cost:.2f} $.")
    else:
        print("Impossible d'estimer le coût pour 500 000 images : aucune segmentation réussie.")
    return batch_segmentations

##########################################################################################################

def display_segmented_images_batch(original_image_paths, segmentation_masks):
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

##########################################################################################################

def colorize_mask(mask, colormap):
    """
    Applique le colormap personnalisé au masque.
    Pour chaque pixel, s'il correspond à un label défini dans colormap,
    la couleur correspondante est assignée.
    """
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in colormap.items():
        colored_mask[mask == label] = color
    return colored_mask

def add_legend(image, legend, start_x=10, start_y=10, box_size=15, spacing=5):
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
        batch_seg_results = segment_images_batch(image_paths)
        print("Traitement en batch terminé.")
    else:
        batch_seg_results = []
        print("Aucune image à traiter en batch.")

    # Afficher les résultats du batch
    print("Affichage des résultats du batch...")
    if batch_seg_results:
        display_segmented_images_batch(image_paths, batch_seg_results)
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
        colored_mask = colorize_mask(msk, custom_colormap)

        # Ajout de la légende sur le masque colorisé
        colored_mask_with_legend = add_legend(colored_mask, legend_labels)

        # Superposition du masque coloré sur l'image originale
        overlay = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)
        overlay_with_legend = add_legend(overlay, legend_labels)

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
