# Playbook — segmentation de vêtements

Ce guide décrit l'exécution du projet tel qu'il est implémenté dans `main.py`,
`segmentation.py`, `visualization.py` et `iou.py`.

## Objectif

À partir d'images de vêtements, le script peut :

1. appeler le modèle de segmentation Hugging Face ou un service local ;
2. enregistrer les masques prédits et des comparaisons visuelles ;
3. calculer l'IoU entre les masques prédits et les masques de référence.

Le mode test permet de ne lancer que l'évaluation IoU sur les masques déjà
présents.

## Prérequis

- Python et les dépendances de `requirements.txt` installées.
- Exécution depuis la racine du dépôt.
- Dossiers présents : `assets/IMG`, `assets/Mask`, `assets/Reference` et
  `resultats_segmentation`. Le script crée `assets/IMG` s'il manque, mais ne
  crée pas les autres dossiers.
- Pour le mode en ligne : un token Hugging Face valide.
- Pour le mode local : le service de segmentation doit répondre à
  `http://localhost:8000/segment` et accepter un envoi multipart avec le champ
  `file`.

## Configuration

Créer `.env` à partir de `.env.example`, puis choisir les valeurs suivantes :

| Variable | Valeur / défaut | Effet |
| --- | --- | --- |
| `token` | Token Hugging Face | Requis pour le mode en ligne. |
| `api_local` | `false` | `true` utilise l'API locale ; `false` utilise Hugging Face. |
| `only_test` | `false` | `true` saute la segmentation et les visualisations, puis lance l'IoU. |
| `max_images` | `3` | Limite le nombre d'images sélectionnées pour la segmentation et la visualisation. |

Exemple pour traiter des images avec Hugging Face :

```dotenv
token=votre_token_huggingface
api_local=false
only_test=false
max_images=3
```

Le token n'est pas nécessaire pour l'API locale. `main.py` affiche néanmoins
un avertissement si `token` est absent, même en mode local.

## Exécution complète

1. Placer les images d'entrée `.jpg` ou `.png` dans `assets/IMG`.
2. Choisir le mode et le nombre d'images avec `.env`.
3. Vérifier la liste des images affichée au démarrage : `main.py` prend les
   premières images renvoyées par le système de fichiers, sans tri garanti.
4. Depuis la racine du dépôt, lancer :

   ```powershell
   python main.py
   ```

5. Vérifier les fichiers générés, les éventuels messages d'erreur, puis le
   bilan IoU affiché dans le terminal.

### Résultats attendus

- Masques prédits : `assets/Mask/mask_<nom_image>`.
- Comparaisons image/masque : `resultats_segmentation/<nom_image>_comparison.png`.
- Montages colorisés avec légende : `resultats_segmentation/result_<index>.png`.
- Mesures IoU par fichier et par classe dans la sortie console.

Les masques utilisent des identifiants de classe de 0 à 17 ; `0` correspond au
fond. L'IoU moyenne hors fond exclut la classe `0`.

## Évaluation IoU seule

Pour évaluer les masques déjà générés, configurer :

```dotenv
only_test=true
```

Puis lancer `python main.py`. La segmentation n'est pas appelée, mais
l'évaluation finale de `assets/Mask` par rapport à `assets/Reference` est tout
de même exécutée.

Pour comparer une seule paire indépendamment de `main.py` :

```powershell
python iou.py assets\Mask\mask_image_0.png assets\Reference\mask_image_0.png
```

Les deux masques doivent avoir les mêmes dimensions et contenir des
identifiants de classe valides. L'évaluation par dossiers exige les mêmes noms
de fichiers dans `assets/Mask` et `assets/Reference` ; elle compare tous les
fichiers présents dans ces dossiers, et non seulement `max_images`.

## Dépannage

- **Aucune image à traiter** : vérifier que `assets/IMG` contient des fichiers
  `.jpg` ou `.png` et que `max_images` est un entier positif.
- **Échec en mode en ligne** : vérifier `token`, l'accès réseau et la réponse
  du service Hugging Face.
- **Échec en mode local** : vérifier que le serveur local est démarré et
  accessible à l'URL configurée dans `main.py`.
- **Évaluation IoU impossible** : vérifier l'existence des dossiers `Mask` et
  `Reference`, l'appariement exact des noms de fichiers et la validité des
  masques.
- **Résultats visuels qui ne correspondent pas aux images** : la création des
  montages prend les images et les masques triés séparément puis les associe
  par position. Éviter les masques anciens ou des listes non alignées avant
  de lancer le script.

## Limites à connaître

- `max_images` borne la sélection des images d'entrée, mais ne borne pas
  l'évaluation IoU par dossiers.
- La sélection de segmentation n'est pas triée ; consulter la liste imprimée
  au lancement pour confirmer les images retenues.
- Les fichiers image et masque utilisés pour les montages sont appariés par
  position après tri, tandis que l'IoU par dossiers apparie les masques par
  nom exact.
- Le script n'enregistre pas de rapport IoU dans un fichier : les scores sont
  affichés dans le terminal.
