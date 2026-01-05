"""
Test du parser VoiceStick avec les fichiers fournis
"""

from pathlib import Path
import sys
sys.path.append('/home/claude')

from voicestick_parser import VoiceStickParser

# Chemins des fichiers de test
main_tg = Path("/mnt/user-data/uploads/01_04_25_11_19_02_000.TextGrid")
commands_tg = Path("/mnt/user-data/uploads/01_04_25_11_19_02_000_commands.TextGrid")
audio_file = "01_04_25_11_19_02_000.wav"

# Créer le parser
parser = VoiceStickParser(tolerance=1.5)

# Traiter la paire de fichiers
print("=== PARSING DES TEXTGRID ===\n")
segments = parser.process_file_pair(main_tg, commands_tg, audio_file)

print(f"Nombre total de segments: {len(segments)}\n")

# Afficher quelques exemples intéressants
print("=== EXEMPLES DE SEGMENTS ===\n")

# Exemple 1: Un segment "droite" (ambiguïté translation/rotation)
print("1. Segments avec 'droite' (résolution d'ambiguïté):")
droite_segs = [s for s in segments if 'droite' in s.transcription]
for seg in droite_segs[:3]:
    print(f"  Transcription: '{seg.transcription}'")
    print(f"  Temps: {seg.start:.2f}s - {seg.end:.2f}s")
    print(f"  Commande finale: {seg.command}")
    print(f"  Confidence joystick: {seg.joystick_match_confidence:.2f}")
    print()

# Exemple 2: Un segment "gauche"
print("\n2. Segments avec 'gauche' (résolution d'ambiguïté):")
gauche_segs = [s for s in segments if 'gauche' in s.transcription]
for seg in gauche_segs[:3]:
    print(f"  Transcription: '{seg.transcription}'")
    print(f"  Temps: {seg.start:.2f}s - {seg.end:.2f}s")
    print(f"  Commande finale: {seg.command}")
    print(f"  Confidence joystick: {seg.joystick_match_confidence:.2f}")
    print()

# Exemple 3: Segments "stop"
print("\n3. Segments 'stop':")
stop_segs = [s for s in segments if 'stop' in s.transcription]
for seg in stop_segs:
    print(f"  Transcription: '{seg.transcription}'")
    print(f"  Temps: {seg.start:.2f}s - {seg.end:.2f}s")
    print(f"  Commande finale: {seg.command}")
    print()

# Exemple 4: Segments "avant/droit"
print("\n4. Segments 'avant/droit' (forward):")
forward_segs = [s for s in segments if 'avant' in s.transcription or 'droit' in s.transcription]
for seg in forward_segs[:3]:
    print(f"  Transcription: '{seg.transcription}'")
    print(f"  Temps: {seg.start:.2f}s - {seg.end:.2f}s")
    print(f"  Commande finale: {seg.command}")
    print(f"  Confidence joystick: {seg.joystick_match_confidence:.2f}")
    print()

# Exemple 5: Segments non-directifs (none)
print("\n5. Segments non-directifs (classe 'none'):")
none_segs = [s for s in segments if s.command == 'none']
for seg in none_segs[:5]:
    print(f"  Transcription: '{seg.transcription}'")
    print(f"  Commande: {seg.command}")
    print()

# Statistiques globales
print("\n=== STATISTIQUES GLOBALES ===\n")

from collections import Counter
command_counts = Counter([s.command for s in segments])

print("Distribution des classes:")
for cmd, count in command_counts.most_common():
    percentage = (count / len(segments)) * 100
    print(f"  {cmd:12s}: {count:3d} segments ({percentage:5.1f}%)")

print(f"\nConfiance moyenne du matching joystick: {sum(s.joystick_match_confidence for s in segments) / len(segments):.2f}")

# Segments avec faible confidence
low_conf = [s for s in segments if s.joystick_match_confidence < 0.5 and s.joystick_match_confidence > 0]
print(f"\nSegments avec faible confidence (<0.5): {len(low_conf)}")

# Segments sans match joystick
no_match = [s for s in segments if s.joystick_match_confidence == 0]
print(f"Segments sans match joystick: {len(no_match)}")
print(f"  (ces segments sont classés uniquement sur la transcription)")
