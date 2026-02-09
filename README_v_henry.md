# Dataset - VoiceStick

Ce dataset contient des enregistrements audio segmentés et transcrits, alignés avec des commandes de manette (joystick) et des données contextuelles (distance à la cible).

## 🛠️ Pipeline de traitement (Data Processing)

Les données ont subi le traitement suivant :

1.  **Segmentation :** Les fichiers `.wav` bruts sont segmentés en portions de parole via **PyAnnote** (`pyannote/segmentation-3.0`).
2.  **Transcription :** Ces segments sont transcrits textuellement avec **Whisper** (`large-v2`).
3.  **Alignement (MFA) :** Un fichier TextGrid est généré, incluant le découpage en phonèmes via le **Montreal Forced Aligner (MFA)**.
4.  **Enrichissement :**
    * Ajout d'une tier `Distance to Target` (distance au moment de l'énoncé).
    * Génération d'un second TextGrid (`_commands`) synchronisé, contenant les impulsions de la manette.

## 📂 Structure des fichiers

Chaque enregistrement suit la convention de nommage : `JJ.MM.YY_HH.MM.SS_00X`

* `JJ.MM.YY` : Date de la passation.
* `HH.MM.SS` : Heure de début du bloc.
* `00X` : Numéro de l'essai dans le bloc (de 0 à 5).

### 1. Fichier Audio (`.wav`)
`JJ.MM.YY_HH.MM.SS_00X.wav`

> [!WARNING]
> **Attention à l'échantillonnage :**
> Les fichiers sont en **48 000 Hz**. Soyez vigilants si vous utilisez des outils automatiques qui attendent souvent du 16 000 Hz.

### 2. Alignement & Données (`.TextGrid`)
`JJ.MM.YY_HH.MM.SS_00X.TextGrid`

Contient les tiers relatives à la parole et au contexte :
* **Text :** Transcription orthographique des énoncés.
* **Text - words :** Découpage temporel par mots.
* **Text - phones :** Découpage temporel par phonèmes.
* **Distance to Target :** Distance à la cible au début de l'énoncé.

### 3. Commandes Manette (`_commands.TextGrid`)
`JJ.MM.YY_HH.MM.SS_00X_commands.TextGrid`

Contient les tiers relatives aux inputs (impulsions) du joystick :
* **Translation :** `forward`, `backward`, `left`, `right`, `up`, `down`.
* **Rotation (Pivot) :** `yawleft`, `yawright`.

---