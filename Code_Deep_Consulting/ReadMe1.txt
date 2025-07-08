# Code_Deep_Consulting_Hackathon_CONIA2025

Ce dépôt contient le code, les données et les résultats générés pour le projet **Code_Deep_Consulting_CONIA2025**, développé dans le cadre du hackathon CONIA 2025. Le projet vise à utiliser des techniques de deep learning pour diagnostiquer les maladies du maïs (Healthy, Maize Lethal Necrosis - MLN, Maize Streak Virus - MSV) à partir d'images.

## Description

- **Code principal** : Le fichier `Code_Deep_Consulting_CONIA2025` contient le script Python qui entraîne un modèle de deep learning pour classer les images de maïs.
- **Dataset** : Un dossier nommé `Data` contient trois sous-dossiers :
  - `HEALTHY` : Images de maïs en bonne santé.
  - `MLN` : Images de maïs affectés par la nécrose létale du maïs.
  - `MSV` : Images de maïs affectés par le virus de la rayure du maïs.
  Chaque sous-dossier contient des images correspondant à la classe respective.
- **Modèle entraîné** : Le fichier `best_model.pth` est le modèle sauvegardé après l'exécution du code, représentant le meilleur modèle obtenu lors de l'entraînement.
- **Graphiques** : Des graphiques générés pendant l'exécution (par exemple, training_metrics.png ou class_distribution.png) sont sauvegardés sous forme d'images.
- **Archive** : Tous les fichiers et dossiers sont compressés dans un fichier ZIP nommé `Code_Deep_Consulting_Hackathon_CONIA2025.zip` pour une distribution facile.

## Prérequis

- **Python 3.x** : Assurez-vous d'avoir Python installé (recommandé : 3.8 ou supérieur).
- **Dépendances** :
  - `torch` (PyTorch) pour le deep learning.
  - `Pillow` pour manipuler les images.
  - `matplotlib` ou `seaborn` pour générer les graphiques .
  Installez-les via :
  ```bash
  pip install torch Pillow matplotlib