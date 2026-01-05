# Parser TextGrid VoiceStick - Audio-to-Command

Ce parser transforme les paires de fichiers TextGrid (transcriptions + commandes joystick) du corpus VoiceStick en un dataset prêt pour l'entraînement de modèles Audio-to-Command.

## 🎯 Objectif

Créer un mapping direct **audio → commande de drone** pour entraîner des modèles de classification (SVM, MLP) avec embeddings wav2vec2, sans passer par une transcription intermédiaire.

## 📋 Fonctionnalités principales

### 1. Parsing automatique des TextGrid
- ✅ Parse le TextGrid principal (transcriptions + distance à cible)
- ✅ Parse le TextGrid `_commands` (impulsions joystick)
- ✅ Synchronisation temporelle automatique (tolérance configurable)

### 2. Résolution d'ambiguïtés
- ✅ **Gauche/Droite** : Distingue translation (`left`/`right`) vs rotation (`yawleft`/`yawright`)
- ✅ **"Encore"** : Gère les commandes de répétition en se basant sur le joystick
- ✅ **Énoncés non-directifs** : Classe automatiquement en `none`

### 3. Classes de commandes
Le parser produit **9 classes** :
- `forward` : avancer
- `back` : reculer
- `left` : translation latérale gauche
- `right` : translation latérale droite
- `up` : monter
- `down` : descendre
- `yawleft` : rotation gauche
- `yawright` : rotation droite
- `none` : énoncés non-directifs (encouragements, hésitations, etc.)
- `stop` : arrêt

## 🚀 Installation

```bash
pip install pandas --break-system-packages
```

Le parser ne nécessite aucune dépendance externe (parsing TextGrid manuel).

## 💻 Utilisation

### Exemple basique

```python
from pathlib import Path
from voicestick_parser import VoiceStickParser

# Créer le parser
parser = VoiceStickParser(tolerance=1.5)  # Fenêtre temporelle en secondes

# Traiter une paire de fichiers
segments = parser.process_file_pair(
    main_tg_path=Path("fichier.TextGrid"),
    commands_tg_path=Path("fichier_commands.TextGrid"),
    audio_file="fichier.wav"
)

# Chaque segment contient:
# - audio_file: nom du fichier audio
# - start, end: timestamps en secondes
# - transcription: texte transcrit
# - command: classe de commande (forward, back, etc.)
# - joystick_match_confidence: 0-1, confiance du matching temporel
# - distance_to_target: distance à la cible (optionnel)
```

### Traiter un corpus complet

```python
# Créer un dataset CSV complet
df = parser.create_dataset(
    textgrid_dir=Path("/path/to/textgrids"),
    output_csv=Path("/path/to/output/dataset.csv"),
    audio_extension='.wav'
)

# Le CSV contient toutes les informations pour l'entraînement
print(df.head())
```

### Structure du CSV de sortie

```
audio_file,start,end,duration,transcription,command,joystick_confidence,distance_to_target
01_04_25_11_19_02_000.wav,35.32,36.50,1.18,"vers ta droite",down,0.57,148.9
01_04_25_11_19_02_000.wav,36.89,37.86,0.97,"tourne légèrement à droite",right,0.95,135.2
...
```

## 🔍 Logique de classification

### Priorités de décision

1. **Si joystick disponible** → Priorité au joystick (source de vérité)
   - Résout l'ambiguïté gauche/droite automatiquement
   - Gère les incohérences transcription/action

2. **Si pas de joystick** → Classification sur transcription
   - Mots-clés spatiaux détectés
   - Heuristiques pour rotation vs translation
   - Énoncés non-directifs → `none`

### Exemples de résolution d'ambiguïté

#### Exemple 1: Rotation détectée
```
Transcription: "tourne légèrement à droite"
Joystick: right (translation)
→ Résultat: right
```

Avec le mot-clé "tourne", si pas de joystick on aurait inféré `yawright`, mais le joystick indique une translation.

#### Exemple 2: Incohérence transcription/joystick
```
Transcription: "vers ta droite" 
Joystick: down
→ Résultat: down (confiance au joystick)
```

Le pilote a mal interprété ou le guide s'est trompé de direction.

#### Exemple 3: "Encore" avec répétition
```
Transcription: "encore un peu à droite"
Joystick: back
→ Résultat: back
```

"Encore" répète la dernière action, qui n'était pas "droite" mais "back".

## ⚙️ Configuration

### Paramètre `tolerance`

```python
parser = VoiceStickParser(tolerance=1.5)
```

- **Rôle** : Fenêtre temporelle (secondes) pour chercher le joystick correspondant
- **Défaut** : 1.5s
- **Impact** :
  - Trop petit → Moins de matches, plus de classification sur transcription seule
  - Trop grand → Risque de matches incorrects

### Ajuster les mots-clés

Vous pouvez modifier `SPATIAL_KEYWORDS` et `NON_DIRECTIVE_KEYWORDS` dans la classe :

```python
parser.SPATIAL_KEYWORDS['forward'].append('allez')
parser.NON_DIRECTIVE_KEYWORDS.append('voilà')
```

## 📊 Analyse des résultats

Le parser fournit des métriques de qualité :

```python
# Confidence moyenne du matching joystick
mean_conf = df['joystick_confidence'].mean()

# Segments sans match joystick
no_match = df[df['joystick_confidence'] == 0]

# Distribution des classes
print(df['command'].value_counts())
```

## 🐛 Cas limites et gestion des erreurs

### Warnings générés

- `Fichier _commands manquant` : Pas de commandes joystick pour ce fichier
- `Tier 'Text' non trouvé` : Structure TextGrid incorrecte
- `Erreur lors du parsing` : Fichier TextGrid corrompu

### Segments sans joystick

Environ 15-20% des segments peuvent ne pas avoir de match joystick :
- Décalage temporel trop important
- Pas d'impulsion joystick (pause, hésitation)
- Segment en début/fin d'enregistrement

Ces segments sont classés uniquement sur la transcription.

## 📈 Statistiques typiques

Sur le corpus VoiceStick complet (~1470 commandes) :
- **Classe dominante** : `none` (~48%, énoncés non-directifs)
- **Classe la plus rare** : `yawleft`/`yawright` (<5%, rotations peu utilisées)
- **Confidence moyenne** : 0.6-0.7 (bon alignement temporel)

### Déséquilibre des classes

Le déséquilibre important (stop ~18%, directions ~5-10% chacune) nécessitera :
- ✅ Pondération des classes lors de l'entraînement
- ✅ Data augmentation (pitch, vitesse, bruit)
- ✅ Validation stratifiée

## 🔧 Intégration dans le pipeline

```
TextGrid parsing → Segmentation audio → Extraction embeddings → Classification
     ↓
   dataset.csv ← Vous êtes ici
     ↓
   Découper les WAV selon start/end
     ↓
   wav2vec2 embeddings (frozen)
     ↓
   SVM / MLP training
```

## 📝 Format des données en sortie

Chaque ligne du CSV correspond à **un segment audio labellisé** prêt pour :

1. **Segmentation** : `audio_file`, `start`, `end` → Découper le WAV
2. **Feature extraction** : Segment audio → Embeddings wav2vec2
3. **Training** : Embeddings + `command` → Entraîner classifieur

## 🎓 Références

- **Corpus VoiceStick** : Henry et al. (2025), PETRA'25
- **wav2vec2-FR-7K-large** : LeBenchmark, modèle français pré-entraîné
- **Cahier des charges** : Projet M2 TAL, Univ. Grenoble Alpes

## ⚠️ Limitations connues

1. **Ambiguïté "gauche/droite" sans joystick** : Par défaut assume translation
2. **Commandes contextuelles** : "encore" sans contexte → `none`
3. **Segmentation Whisper** : ~5% d'erreurs héritées de la segmentation automatique
4. **Généralisation** : Seulement 20 locuteurs dans le corpus

## 📞 Support

Pour toute question sur l'utilisation du parser :
- Vérifier les exemples dans `test_parser.py` et `demo_parser_detailed.py`
- Consulter le cahier des charges du projet
