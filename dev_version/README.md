# VoiceStick — Guide developpeur

Pipeline complet pour entrainer des classifieurs de commandes de drone a partir d'enregistrements audio annotes. Comprend la preparation des donnees, l'entrainement (SVM et MLP), la prediction et l'evaluation.

## Demarrage rapide

1. Placez vos fichiers `.TextGrid` dans `data/textgrid/` et les fichiers `.wav` correspondants dans `data/audio/`
2. Preparez les donnees :
   ```bash
   python prepare_data.py
   ```
3. Entrainez les modeles :
   ```bash
   python train_svm.py
   python train_mlp.py
   ```
4. Predisez sur l'audio de test :
   ```bash
   python predict.py --use-ground-truth
   ```
5. Evaluez les predictions :
   ```bash
   python evaluate.py
   ```

## Structure du dossier

### Avant execution

```
dev_version/
├── prepare_data.py      # Preparation des donnees
├── train_svm.py         # Entrainement SVM
├── train_mlp.py         # Entrainement MLP
├── predict.py           # Prediction sur audio brut
├── evaluate.py          # Evaluation des predictions
├── config.yaml          # Configuration
├── requirements.txt     # Dependances Python
├── data/
│   ├── textgrid/        # Fichiers .TextGrid (annotations Praat)
│   └── audio/           # Fichiers .wav (enregistrements)
└── output/              # Vide au depart
```

### Apres execution complete

```
output/
├── dataset.csv                  # Dataset complet (tous les participants)
├── train/
│   ├── dataset_train.csv        # Donnees d'entrainement
│   ├── audio_segments/          # Segments audio 16 kHz
│   └── all_embeddings.npz       # Embeddings wav2vec2
├── test/
│   ├── dataset_test.csv         # Labels ground truth (test)
│   └── audio/                   # Fichiers audio bruts (test)
├── SVM_model/
│   ├── model_svm.pkl            # Modele entraine
│   ├── scaler.pkl               # StandardScaler
│   ├── label_encoder.pkl        # Encodeur de labels
│   ├── cv_results_svm.json      # Resultats cross-validation
│   ├── confusion_matrix_svm.png # Matrice de confusion
│   └── svm_output.txt           # Log d'entrainement
├── MLP_model/
│   ├── model_mlp.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── cv_results_mlp.json
│   ├── confusion_matrix_mlp.png
│   └── mlp_output.txt
├── predictions/
│   ├── predictions_svm.csv      # Predictions SVM
│   └── predictions_mlp.csv      # Predictions MLP
└── evaluation/
    ├── eval_svm.json            # Metriques SVM
    ├── eval_mlp.json            # Metriques MLP
    ├── comparison.json          # Comparaison cote a cote
    ├── confusion_matrix_eval_svm.png
    ├── confusion_matrix_eval_mlp.png
    └── evaluation_output.txt
```

## Format des donnees

### Fichiers audio

- Format : `.wav`
- Frequence d'echantillonnage : quelconque (reenregistre a 16 kHz en interne)
- Convention de nommage : `DD_MM_YY_HH_MM_SS_NNN.wav`
  - `DD_MM_YY_HH_MM_SS` : identifiant du participant (date/heure de la session)
  - `NNN` : numero de tentative (`000` a `005`)

### Fichiers TextGrid (Praat)

- Encodage : UTF-8 ou UTF-16 avec BOM (detection automatique)
- Un tier nomme `commands` (insensible a la casse)
- Labels : les 9 classes de commandes

```
forward | backward | left | right | up | down | yawleft | yawright | none
```

Les intervalles non etiquetes ou vides sont automatiquement classes comme `none`.

## Configuration (config.yaml)

```yaml
paths:
  textgrid_dir: "data/textgrid"    # Annotations Praat (.TextGrid)
  audio_dir: "data/audio"          # Enregistrements audio (.wav)
  output_dir: "output"             # Sorties du pipeline

data_preparation:
  tier_name: "commands"    # Nom du tier TextGrid
  skip_if_cached: true     # Reutiliser les fichiers intermediaires existants
  test_size: 0.15          # Proportion de participants reserves pour le test
  random_seed: 42          # Graine pour la reproductibilite du split

training:
  balance_classes: true    # Sous-echantillonner la classe "none"
  none_ratio: 1.0          # Ratio max de "none" vs la plus grande classe de commande
  n_folds: 5               # Nombre de folds pour la cross-validation (GroupKFold)
```

## Etapes du pipeline

### 1. Preparation des donnees — `python prepare_data.py`

Cinq etapes executees sequentiellement :

1. **Parsing** des TextGrids : extrait les annotations du tier `commands` vers `dataset.csv`
2. **Split** des participants 85/15 en ensembles train/test (par participant, pas par segment)
3. **Segmentation** de l'audio d'entrainement en clips WAV 16 kHz dans `train/audio_segments/`
4. **Extraction d'embeddings** wav2vec2-FR-7K-large (mean-pooling) vers `train/all_embeddings.npz`
5. **Copie** des fichiers audio bruts des participants de test vers `test/audio/`

L'option `skip_if_cached: true` permet de reprendre une execution interrompue sans recalculer les etapes deja completees.

### 2. Entrainement SVM — `python train_svm.py`

- Lit les donnees depuis `output/train/`
- Cross-validation GroupKFold a 5 folds (groupement par participant)
- Par fold : equilibrage des classes, StandardScaler, SVC(kernel='rbf', C=10, class_weight='balanced')
- Affiche les metriques par fold et la moyenne
- Entraine le modele final sur toutes les donnees d'entrainement
- Sauvegarde dans `output/SVM_model/`

### 3. Entrainement MLP — `python train_mlp.py`

Meme structure que le SVM :

- Cross-validation GroupKFold a 5 folds
- Par fold : equilibrage des classes, StandardScaler, MLPClassifier
- Sauvegarde dans `output/MLP_model/`

### 4. Prediction — `python predict.py --use-ground-truth`

En mode `--use-ground-truth` (recommande pour l'evaluation) :

- Utilise les bornes temporelles de `dataset_test.csv`
- Extrait les embeddings wav2vec2 des segments de test
- Classifie avec SVM et/ou MLP
- Sauvegarde les predictions CSV dans `output/predictions/`

Autres modes disponibles :

```bash
python predict.py                          # VAD sur test/audio/, les deux modeles
python predict.py --model svm              # SVM uniquement
python predict.py --audio path/to/file.wav # N'importe quel fichier audio
```

### 5. Evaluation — `python evaluate.py`

- Compare les predictions (`predictions_svm.csv`, `predictions_mlp.csv`) aux labels ground truth
- Par modele : accuracy, F1-macro, F1-weighted, rapport par classe, matrice de confusion
- Tableau de comparaison cote a cote si les deux modeles sont disponibles
- Sauvegarde dans `output/evaluation/`

## Installation

```bash
pip install -r requirements.txt
```

Necessite Python 3.10+. Le premier lancement de `prepare_data.py` telecharge automatiquement le modele wav2vec2 (~3 Go).

---

# English Version

# VoiceStick — Developer Guide

Full pipeline for training drone command classifiers from annotated audio recordings. Includes data preparation, training (SVM and MLP), prediction, and evaluation.

## Quick start

1. Place your `.TextGrid` files in `data/textgrid/` and matching `.wav` files in `data/audio/`
2. Prepare the data:
   ```bash
   python prepare_data.py
   ```
3. Train the models:
   ```bash
   python train_svm.py
   python train_mlp.py
   ```
4. Predict on test audio:
   ```bash
   python predict.py --use-ground-truth
   ```
5. Evaluate predictions:
   ```bash
   python evaluate.py
   ```

## Folder structure

### Before running

```
dev_version/
├── prepare_data.py      # Data preparation
├── train_svm.py         # SVM training
├── train_mlp.py         # MLP training
├── predict.py           # Prediction on raw audio
├── evaluate.py          # Prediction evaluation
├── config.yaml          # Configuration
├── requirements.txt     # Python dependencies
├── data/
│   ├── textgrid/        # .TextGrid files (Praat annotations)
│   └── audio/           # .wav files (recordings)
└── output/              # Empty initially
```

### After full pipeline run

```
output/
├── dataset.csv                  # Full dataset (all participants)
├── train/
│   ├── dataset_train.csv        # Training data
│   ├── audio_segments/          # 16 kHz audio segments
│   └── all_embeddings.npz       # wav2vec2 embeddings
├── test/
│   ├── dataset_test.csv         # Ground truth labels (test)
│   └── audio/                   # Raw audio files (test)
├── SVM_model/
│   ├── model_svm.pkl            # Trained model
│   ├── scaler.pkl               # StandardScaler
│   ├── label_encoder.pkl        # Label encoder
│   ├── cv_results_svm.json      # Cross-validation results
│   ├── confusion_matrix_svm.png # Confusion matrix
│   └── svm_output.txt           # Training log
├── MLP_model/
│   ├── model_mlp.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── cv_results_mlp.json
│   ├── confusion_matrix_mlp.png
│   └── mlp_output.txt
├── predictions/
│   ├── predictions_svm.csv      # SVM predictions
│   └── predictions_mlp.csv      # MLP predictions
└── evaluation/
    ├── eval_svm.json            # SVM metrics
    ├── eval_mlp.json            # MLP metrics
    ├── comparison.json          # Side-by-side comparison
    ├── confusion_matrix_eval_svm.png
    ├── confusion_matrix_eval_mlp.png
    └── evaluation_output.txt
```

## Data format

### Audio files

- Format: `.wav`
- Sample rate: any (resampled to 16 kHz internally)
- Naming convention: `DD_MM_YY_HH_MM_SS_NNN.wav`
  - `DD_MM_YY_HH_MM_SS`: participant identifier (session date/time)
  - `NNN`: attempt number (`000` to `005`)

### TextGrid files (Praat)

- Encoding: UTF-8 or UTF-16 with BOM (auto-detected)
- A tier named `commands` (case-insensitive)
- Labels: the 9 command classes

```
forward | backward | left | right | up | down | yawleft | yawright | none
```

Unlabeled or empty intervals are automatically classified as `none`.

## Configuration (config.yaml)

```yaml
paths:
  textgrid_dir: "data/textgrid"    # Praat annotations (.TextGrid)
  audio_dir: "data/audio"          # Audio recordings (.wav)
  output_dir: "output"             # Pipeline outputs

data_preparation:
  tier_name: "commands"    # TextGrid tier name
  skip_if_cached: true     # Reuse existing intermediate files
  test_size: 0.15          # Fraction of participants held out for testing
  random_seed: 42          # Seed for reproducible train/test split

training:
  balance_classes: true    # Subsample the "none" class
  none_ratio: 1.0          # Max ratio of "none" vs largest command class
  n_folds: 5               # Number of cross-validation folds (GroupKFold)
```

## Pipeline steps

### 1. Data preparation — `python prepare_data.py`

Five steps executed sequentially:

1. **Parsing** TextGrids: extracts annotations from the `commands` tier into `dataset.csv`
2. **Splitting** participants 85/15 into train/test sets (by participant, not by segment)
3. **Segmenting** training audio into 16 kHz WAV clips in `train/audio_segments/`
4. **Extracting** wav2vec2-FR-7K-large embeddings (mean-pooling) into `train/all_embeddings.npz`
5. **Copying** raw audio files for test participants into `test/audio/`

The `skip_if_cached: true` option allows resuming an interrupted run without recomputing completed steps.

### 2. SVM training — `python train_svm.py`

- Reads data from `output/train/`
- 5-fold GroupKFold cross-validation (grouped by participant)
- Per fold: class balancing, StandardScaler, SVC(kernel='rbf', C=10, class_weight='balanced')
- Reports per-fold and average F1 metrics
- Trains final model on all training data
- Saves to `output/SVM_model/`

### 3. MLP training — `python train_mlp.py`

Same structure as SVM:

- 5-fold GroupKFold cross-validation
- Per fold: class balancing, StandardScaler, MLPClassifier
- Saves to `output/MLP_model/`

### 4. Prediction — `python predict.py --use-ground-truth`

In `--use-ground-truth` mode (recommended for evaluation):

- Uses segment boundaries from `dataset_test.csv`
- Extracts wav2vec2 embeddings from test segments
- Classifies with SVM and/or MLP
- Saves prediction CSVs to `output/predictions/`

Other available modes:

```bash
python predict.py                          # VAD on test/audio/, both models
python predict.py --model svm              # SVM only
python predict.py --audio path/to/file.wav # Any audio file
```

### 5. Evaluation — `python evaluate.py`

- Compares predictions (`predictions_svm.csv`, `predictions_mlp.csv`) against ground truth labels
- Per model: accuracy, F1-macro, F1-weighted, per-class report, confusion matrix
- Side-by-side comparison table when both models are available
- Saves to `output/evaluation/`

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. The first run of `prepare_data.py` automatically downloads the wav2vec2 model (~3 GB).
