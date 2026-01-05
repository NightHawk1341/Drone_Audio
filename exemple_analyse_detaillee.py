"""
Exemple d'utilisation du parser VoiceStick
Avec analyse détaillée du mapping transcription -> commande
"""

from pathlib import Path
import sys
sys.path.append('/home/claude')

from voicestick_parser import VoiceStickParser
import pandas as pd

# Chemins des fichiers de test
main_tg = Path("/mnt/user-data/uploads/01_04_25_11_19_02_000.TextGrid")
commands_tg = Path("/mnt/user-data/uploads/01_04_25_11_19_02_000_commands.TextGrid")
audio_file = "01_04_25_11_19_02_000.wav"

# Créer le parser
parser = VoiceStickParser(tolerance=1.5)

# Traiter la paire de fichiers
print("=" * 80)
print("ANALYSE DÉTAILLÉE DU MAPPING TRANSCRIPTION -> COMMANDE")
print("=" * 80)
print()

segments = parser.process_file_pair(main_tg, commands_tg, audio_file)

# Convertir en DataFrame pour faciliter l'analyse
df = pd.DataFrame([
    {
        'start': s.start,
        'end': s.end,
        'transcription': s.transcription,
        'command': s.command,
        'joystick_conf': s.joystick_match_confidence
    }
    for s in segments
])

print(f"Total de segments: {len(df)}")
print()

# Afficher tous les segments avec détails
print("LISTE COMPLÈTE DES SEGMENTS")
print("-" * 80)
for idx, row in df.iterrows():
    print(f"\n[{idx+1}] {row['start']:.2f}s - {row['end']:.2f}s")
    print(f"    Transcription: \"{row['transcription']}\"")
    print(f"    → Commande: {row['command']}")
    
    if row['joystick_conf'] > 0:
        conf_emoji = "✓" if row['joystick_conf'] > 0.7 else "~"
        print(f"    Joystick match: {conf_emoji} {row['joystick_conf']:.2f}")
    else:
        print(f"    Joystick match: ✗ (classification sur transcription seule)")

print()
print("=" * 80)
print("ANALYSE PAR CLASSE DE COMMANDE")
print("=" * 80)
print()

for cmd in df['command'].unique():
    cmd_df = df[df['command'] == cmd]
    print(f"\n{cmd.upper()} ({len(cmd_df)} segments)")
    print("-" * 40)
    for _, row in cmd_df.iterrows():
        print(f"  • \"{row['transcription']}\" [{row['start']:.1f}s]")

print()
print("=" * 80)
print("CAS INTÉRESSANTS")
print("=" * 80)

# Cas 1: Résolution ambiguïté gauche/droite
print("\n1. RÉSOLUTION AMBIGUÏTÉ GAUCHE/DROITE")
print("-" * 40)
ambiguous = df[df['transcription'].str.contains('gauche|droite', case=False)]
for _, row in ambiguous.iterrows():
    joy_type = "translation" if row['command'] in ['left', 'right'] else "rotation" if row['command'] in ['yawleft', 'yawright'] else "autre"
    print(f"  \"{row['transcription']}\"")
    print(f"  → {row['command']} ({joy_type})")
    print()

# Cas 2: Segments "encore"
print("\n2. GESTION DE 'ENCORE' (répétition)")
print("-" * 40)
encore = df[df['transcription'].str.contains('encore', case=False)]
if len(encore) > 0:
    for _, row in encore.iterrows():
        print(f"  \"{row['transcription']}\"")
        print(f"  → {row['command']} (joystick conf: {row['joystick_conf']:.2f})")
        print()
else:
    print("  Aucun segment 'encore' dans cet exemple")

# Cas 3: Segments avec faible confidence de matching
print("\n3. SEGMENTS AVEC DÉSACCORD TRANSCRIPTION/JOYSTICK")
print("-" * 40)
print("(Quand la transcription dit une chose mais le joystick en fait une autre)")
print()

for _, row in df.iterrows():
    trans = row['transcription']
    cmd = row['command']
    conf = row['joystick_conf']
    
    # Détecter les désaccords
    if conf > 0:  # Il y a un joystick
        # Vérifier si la transcription suggère une direction différente
        if 'avant' in trans or 'avance' in trans or 'droit' in trans:
            if cmd not in ['forward', 'none']:
                print(f"  DÉSACCORD: \"{trans}\"")
                print(f"  Transcription suggère: forward, Joystick dit: {cmd}")
                print(f"  → Décision: {cmd} (confiance au joystick)")
                print()
        
        elif 'droite' in trans:
            if cmd not in ['right', 'yawright', 'none']:
                print(f"  DÉSACCORD: \"{trans}\"")
                print(f"  Transcription suggère: right/yawright, Joystick dit: {cmd}")
                print(f"  → Décision: {cmd}")
                print()
        
        elif 'gauche' in trans:
            if cmd not in ['left', 'yawleft', 'none']:
                print(f"  DÉSACCORD: \"{trans}\"")
                print(f"  Transcription suggère: left/yawleft, Joystick dit: {cmd}")
                print(f"  → Décision: {cmd}")
                print()

print()
print("=" * 80)
print("STATISTIQUES FINALES")
print("=" * 80)
print()

print(f"Distribution des classes:")
for cmd, count in df['command'].value_counts().items():
    pct = (count / len(df)) * 100
    bar = "█" * int(pct / 2)
    print(f"  {cmd:12s}: {count:2d} ({pct:5.1f}%) {bar}")

print(f"\nConfiance moyenne du matching joystick: {df['joystick_conf'].mean():.2f}")
print(f"Segments avec joystick match: {len(df[df['joystick_conf'] > 0])}/{len(df)}")
print(f"Segments classés sur transcription seule: {len(df[df['joystick_conf'] == 0])}/{len(df)}")

print()
print("=" * 80)
