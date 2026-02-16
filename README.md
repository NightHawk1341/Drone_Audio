# VoiceStick — Classification de commandes vocales pour drone

Pipeline de classification parole-vers-commande pour le pilotage de drones. Des enregistrements audio de locuteurs francophones prononçant des commandes de pilotage sont convertis en embeddings wav2vec2, puis classifiés par des modeles SVM ou MLP.

## Classes de commandes

Les 9 classes reconnues :

| Commande | Description |
|---|---|
| `forward` | Avancer |
| `backward` | Reculer |
| `left` | Translation a gauche |
| `right` | Translation a droite |
| `up` | Monter |
| `down` | Descendre |
| `yawleft` | Rotation a gauche (lacet) |
| `yawright` | Rotation a droite (lacet) |
| `none` | Pas de commande / commande non-explicite |

## Contenu du depot

| Dossier | Description | Usage |
|---|---|---|
| `user_version/` | Package d'inference avec modeles pre-entraines | Utilisateurs finaux qui veulent simplement obtenir des predictions sur leurs fichiers audio |
| `dev_version/` | Pipeline complet d'entrainement | Developpeurs qui veulent entrainer leurs propres modeles sur de nouvelles donnees |

Chaque dossier contient son propre `README.md` avec les instructions detaillees.

## Performance des modeles (jeu de test, 9 participants)

| Metrique | SVM | MLP |
|---|---|---|
| Accuracy | 0.836 | **0.866** |
| F1-macro | 0.735 | **0.774** |
| F1-weighted | 0.843 | **0.868** |

## Stack technique

- **Embeddings** : [wav2vec2-FR-7K-large](https://huggingface.co/LeBenchmark/wav2vec2-FR-7K-large) (LeBenchmark) — embeddings de 1024 dimensions
- **Classifieurs** : SVM (noyau RBF) et MLP (scikit-learn)
- **Segmentation** : VAD par energie via librosa (`librosa.effects.split`)
- **Validation croisee** : GroupKFold a 5 folds par participant (independance locuteur)

## Installation

```bash
pip install -r requirements.txt
```

## Credits

- Modele wav2vec2 : [LeBenchmark](https://huggingface.co/LeBenchmark) — wav2vec2-FR-7K-large
- Classification : [scikit-learn](https://scikit-learn.org/)
- Traitement audio : [librosa](https://librosa.org/)

---

# English Version

# VoiceStick — Drone Voice Command Classification

Speech-to-command classification pipeline for drone piloting. Audio recordings of French speakers issuing piloting commands are converted into wav2vec2 embeddings, then classified using SVM or MLP models.

## Command classes

The 9 recognized classes:

| Command | Description |
|---|---|
| `forward` | Move forward |
| `backward` | Move backward |
| `left` | Strafe left |
| `right` | Strafe right |
| `up` | Ascend |
| `down` | Descend |
| `yawleft` | Rotate left (yaw) |
| `yawright` | Rotate right (yaw) |
| `none` | No command / non-explicit command |

## Repository contents

| Folder | Description | Use case |
|---|---|---|
| `user_version/` | Inference package with pre-trained models | End users who simply want predictions on their audio files |
| `dev_version/` | Full training pipeline | Developers who want to train their own models on new data |

Each folder contains its own `README.md` with detailed instructions.

## Model performance (test set, 9 participants)

| Metric | SVM | MLP |
|---|---|---|
| Accuracy | 0.836 | **0.866** |
| F1-macro | 0.735 | **0.774** |
| F1-weighted | 0.843 | **0.868** |

## Tech stack

- **Embeddings**: [wav2vec2-FR-7K-large](https://huggingface.co/LeBenchmark/wav2vec2-FR-7K-large) (LeBenchmark) — 1024-dimensional embeddings
- **Classifiers**: SVM (RBF kernel) and MLP (scikit-learn)
- **Segmentation**: Energy-based VAD via librosa (`librosa.effects.split`)
- **Cross-validation**: 5-fold GroupKFold by participant (speaker-independent)

## Installation

```bash
pip install -r requirements.txt
```

## Credits

- wav2vec2 model: [LeBenchmark](https://huggingface.co/LeBenchmark) — wav2vec2-FR-7K-large
- Classification: [scikit-learn](https://scikit-learn.org/)
- Audio processing: [librosa](https://librosa.org/)
