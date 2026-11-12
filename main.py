
import base64
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# TODO: Modifiez ces valeurs selon votre configuration
image_dir = "assets/IMG"  # Exemple : si vous êtes sur Colab et avez uploadé un dossier
max_images = 1  # Commençons avec peu d'images

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
if api_local:
    API_URL = API_URL_LOCAL
print(f"Mode : {'local' if api_local else 'online'} -> {API_URL}")

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

if image_paths:
    single_image_path = image_paths[0] # première image de la liste
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
                API_URL,
                files={"file": (os.path.basename(single_image_path), image_data, content_type)},
                timeout=120,
            )
        else:
            response = requests.post(
                API_URL,
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
        segmentation_mask_image.save("assets/Mask/segmentation_mask_image.png")
        print("Masque sauvegardé sous 'segmentation_mask_image.png'.")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")
else:
    print("Aucune image à traiter. Vérifiez la configuration de 'image_dir' et 'max_images'.")