"""
Parser TextGrid pour le corpus VoiceStick
Mapping Audio-to-Command pour l'entraînement de modèles

Ce parser synchronise les transcriptions avec les impulsions joystick réelles
pour créer un dataset d'entraînement audio -> commande drone
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import re


@dataclass
class AudioSegment:
    """Représente un segment audio avec sa commande associée"""
    audio_file: str
    start: float
    end: float
    duration: float
    transcription: str
    command: str  # forward, back, left, right, up, down, yawleft, yawright, none
    joystick_match_confidence: float  # 0-1, basé sur la distance temporelle
    distance_to_target: Optional[float] = None


class VoiceStickParser:
    """
    Parser pour créer un dataset Audio-to-Command à partir des TextGrid VoiceStick
    """
    
    # Mapping des mots-clés vers les axes spatiaux (avant résolution d'ambiguïté)
    SPATIAL_KEYWORDS = {
        'forward': ['avant', 'avance', 'forward', 'devant', 'face', 'droit'],
        'back': ['arrière', 'recule', 'back', 'derrière', 'recul'],
        'up': ['haut', 'monte', 'up', 'dessus', 'lève', 'lever'],
        'down': ['bas', 'descend', 'down', 'dessous', 'baisse', 'baisser'],
        'left_or_right': ['gauche', 'droite', 'left', 'right'],  # Ambiguë !
        'stop': ['stop', 'arrête', 'arrete', 'stoppe', 'ok stop']
    }
    
    # Énoncés considérés comme non-directifs (classe "none")
    NON_DIRECTIVE_KEYWORDS = [
        'encore', 'ok', 'vas-y', 'doucement', 'c\'est bien', 'voilà', 
        'parfait', 'ici', 'ouais', 'attends', 'mince', 'ah', 'là',
        'peut-être', 'peut être', 'c\'est ça', 't es dessus'
    ]
    
    # Commandes possibles dans le fichier _commands.TextGrid
    JOYSTICK_COMMANDS = ['forward', 'backward', 'left', 'right', 'up', 'down', 'yawleft', 'yawright']
    
    def __init__(self, tolerance: float = 1.5):
        """
        Args:
            tolerance: Fenêtre temporelle (en secondes) pour chercher 
                      la commande joystick correspondante
        """
        self.tolerance = tolerance
    
    def _parse_textgrid_manual(self, tg_path: Path) -> Dict:
        """
        Parse manuellement un fichier TextGrid
        
        Returns:
            Dict avec 'tiers' contenant une liste de tiers, 
            chaque tier ayant 'name' et 'intervals'
        """
        with open(tg_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        tiers = []
        current_tier = None
        current_interval = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Début d'un nouveau tier
            if line.startswith('class = "IntervalTier"'):
                if current_tier:
                    tiers.append(current_tier)
                current_tier = {'name': None, 'intervals': []}
            
            # Nom du tier
            elif line.startswith('name = '):
                name = line.split('=')[1].strip().strip('"')
                if current_tier is not None:
                    current_tier['name'] = name
            
            # Début d'un interval
            elif line.startswith('xmin =') and 'intervals:' not in lines[max(0, i-5):i]:
                if current_interval:
                    current_tier['intervals'].append(current_interval)
                
                xmin = float(line.split('=')[1].strip())
                current_interval = {'xmin': xmin, 'xmax': None, 'text': ''}
            
            # xmax de l'interval
            elif line.startswith('xmax =') and current_interval is not None:
                xmax = float(line.split('=')[1].strip())
                current_interval['xmax'] = xmax
            
            # Texte de l'interval
            elif line.startswith('text = ') and current_interval is not None:
                text = line.split('=', 1)[1].strip().strip('"')
                current_interval['text'] = text
            
            i += 1
        
        # Ajouter le dernier interval et tier
        if current_interval:
            current_tier['intervals'].append(current_interval)
        if current_tier:
            tiers.append(current_tier)
        
        return {'tiers': tiers}
        
    def parse_main_textgrid(self, tg_path: Path) -> List[Dict]:
        """
        Parse le TextGrid principal contenant les transcriptions
        
        Returns:
            Liste de dictionnaires avec start, end, transcription, distance
        """
        try:
            tg = self._parse_textgrid_manual(tg_path)
        except Exception as e:
            warnings.warn(f"Erreur lors du parsing de {tg_path}: {e}")
            return []
        
        segments = []
        
        # Trouver le tier "Text" et "Distance to Target"
        text_tier = None
        distance_tier = None
        
        for tier in tg['tiers']:
            if tier['name'] == 'Text':
                text_tier = tier
            elif tier['name'] == 'Distance to Target':
                distance_tier = tier
        
        if not text_tier:
            warnings.warn(f"Tier 'Text' non trouvé dans {tg_path}")
            return []
        
        # Créer un mapping distance par timestamp
        distance_map = {}
        if distance_tier:
            for interval in distance_tier['intervals']:
                if interval['text'].strip():
                    key = f"{interval['xmin']:.3f}"
                    distance_map[key] = interval['text'].strip()
        
        # Parser les segments de texte
        for interval in text_tier['intervals']:
            transcription = interval['text'].strip()
            
            # Ignorer les segments vides
            if not transcription:
                continue
                
            segment = {
                'start': interval['xmin'],
                'end': interval['xmax'],
                'transcription': transcription.lower(),
                'distance': None
            }
            
            # Chercher la distance correspondante
            key = f"{interval['xmin']:.3f}"
            if key in distance_map:
                try:
                    segment['distance'] = float(distance_map[key])
                except ValueError:
                    pass
            
            segments.append(segment)
        
        return segments
    
    def parse_commands_textgrid(self, cmd_tg_path: Path) -> List[Dict]:
        """
        Parse le TextGrid _commands contenant les impulsions joystick
        
        Returns:
            Liste de dictionnaires avec start, end, command
        """
        try:
            tg = self._parse_textgrid_manual(cmd_tg_path)
        except Exception as e:
            warnings.warn(f"Erreur lors du parsing de {cmd_tg_path}: {e}")
            return []
        
        commands = []
        
        # Parcourir tous les tiers (forward, backward, left, right, etc.)
        for tier in tg['tiers']:
            tier_name = tier['name'].lower()
            
            # Vérifier que c'est un tier de commande valide
            if tier_name not in self.JOYSTICK_COMMANDS:
                continue
            
            for interval in tier['intervals']:
                # Ignorer les intervalles vides
                if not interval['text'].strip():
                    continue
                
                commands.append({
                    'start': interval['xmin'],
                    'end': interval['xmax'],
                    'command': tier_name,
                    'duration': interval['xmax'] - interval['xmin']
                })
        
        # Trier par timestamp
        commands.sort(key=lambda x: x['start'])
        return commands
    
    def find_matching_command(
        self, 
        segment_start: float, 
        segment_end: float, 
        commands: List[Dict]
    ) -> Tuple[Optional[str], float]:
        """
        Trouve la commande joystick qui correspond temporellement au segment audio
        
        Args:
            segment_start: Début du segment audio
            segment_end: Fin du segment audio
            commands: Liste des commandes joystick
            
        Returns:
            (commande, confidence) où confidence est entre 0 et 1
            Si aucune commande trouvée, retourne (None, 0.0)
        """
        segment_center = (segment_start + segment_end) / 2
        
        # Chercher les commandes dans la fenêtre temporelle
        candidates = []
        for cmd in commands:
            cmd_center = (cmd['start'] + cmd['end']) / 2
            time_diff = abs(cmd_center - segment_center)
            
            if time_diff <= self.tolerance:
                # Confidence basée sur la proximité temporelle
                confidence = 1.0 - (time_diff / self.tolerance)
                candidates.append((cmd, time_diff, confidence))
        
        if not candidates:
            return None, 0.0
        
        # Prendre la commande la plus proche temporellement
        best_match = min(candidates, key=lambda x: x[1])
        return best_match[0]['command'], best_match[2]
    
    def classify_segment(
        self, 
        transcription: str, 
        joystick_command: Optional[str] = None
    ) -> str:
        """
        Classifie un segment en fonction de la transcription ET de la commande joystick
        
        Cette méthode résout notamment l'ambiguïté gauche/droite entre translation et rotation
        
        Args:
            transcription: Texte transcrit
            joystick_command: Commande joystick réelle (ou None si non trouvée)
            
        Returns:
            Classe de commande: forward, back, left, right, up, down, 
                               yawleft, yawright, stop, ou none
        """
        trans_lower = transcription.lower()
        
        # 1. Si on a une commande joystick avec bonne confidence, on lui fait prioritairement confiance
        if joystick_command:
            joy_normalized = self._normalize_command(joystick_command)
            
            # Vérifier si la transcription contient des mots clés cohérents
            has_forward_kw = any(kw in trans_lower for kw in self.SPATIAL_KEYWORDS['forward'])
            has_back_kw = any(kw in trans_lower for kw in self.SPATIAL_KEYWORDS['back'])
            has_up_kw = any(kw in trans_lower for kw in self.SPATIAL_KEYWORDS['up'])
            has_down_kw = any(kw in trans_lower for kw in self.SPATIAL_KEYWORDS['down'])
            has_left_kw = 'gauche' in trans_lower or 'left' in trans_lower
            has_right_kw = 'droite' in trans_lower or 'right' in trans_lower
            has_stop_kw = any(kw in trans_lower for kw in self.SPATIAL_KEYWORDS['stop'])
            
            # Pour gauche/droite, le joystick résout l'ambiguïté translation/rotation
            if has_left_kw:
                if joy_normalized in ['left', 'yawleft']:
                    return joy_normalized
                # Si joystick dit autre chose mais on dit "gauche", possible incohérence
                # Vérifier si c'est "encore ... gauche" = répéter dernière commande
                if 'encore' in trans_lower:
                    return joy_normalized  # Faire confiance au joystick
                # Sinon garder le joystick
                return joy_normalized
            
            if has_right_kw:
                if joy_normalized in ['right', 'yawright']:
                    return joy_normalized
                if 'encore' in trans_lower:
                    return joy_normalized
                return joy_normalized
            
            # Pour les autres directions, vérifier cohérence
            if has_forward_kw and joy_normalized == 'forward':
                return 'forward'
            if has_back_kw and joy_normalized == 'back':
                return 'back'
            if has_up_kw and joy_normalized == 'up':
                return 'up'
            if has_down_kw and joy_normalized == 'down':
                return 'down'
            if has_stop_kw:
                return 'stop'
            
            # Si aucune cohérence directe, faire confiance au joystick
            # mais vérifier si c'est un énoncé non-directif
            if any(kw in trans_lower for kw in self.NON_DIRECTIVE_KEYWORDS):
                # Énoncé non-directif explicite, ignorer joystick potentiellement erroné
                if 'encore' not in trans_lower:  # Sauf "encore" qui répète
                    return 'none'
            
            return joy_normalized
        
        # 2. Pas de joystick : classifier uniquement sur transcription
        
        # 2a. Énoncés non-directifs
        if any(kw in trans_lower for kw in self.NON_DIRECTIVE_KEYWORDS):
            # Exception: "encore" avec une direction est une commande
            if 'encore' in trans_lower:
                # Si "encore" + direction, classifier selon la direction
                if any(kw in trans_lower for kw in ['gauche', 'droite', 'left', 'right']):
                    # Pas de joystick pour résoudre, mettre en none
                    return 'none'
            return 'none'
        
        # 2b. Stop
        if any(kw in trans_lower for kw in self.SPATIAL_KEYWORDS['stop']):
            return 'stop'
        
        # 2c. Forward
        if any(kw in trans_lower for kw in self.SPATIAL_KEYWORDS['forward']):
            return 'forward'
        
        # 2d. Back
        if any(kw in trans_lower for kw in self.SPATIAL_KEYWORDS['back']):
            return 'back'
        
        # 2e. Up
        if any(kw in trans_lower for kw in self.SPATIAL_KEYWORDS['up']):
            return 'up'
        
        # 2f. Down
        if any(kw in trans_lower for kw in self.SPATIAL_KEYWORDS['down']):
            return 'down'
        
        # 2g. Gauche/Droite ambigu sans joystick
        if 'gauche' in trans_lower or 'left' in trans_lower:
            # Chercher des indices de rotation
            if 'tourne' in trans_lower or 'rotation' in trans_lower or 'pivote' in trans_lower:
                return 'yawleft'
            # Par défaut: translation
            return 'left'
        
        if 'droite' in trans_lower or 'right' in trans_lower:
            if 'tourne' in trans_lower or 'rotation' in trans_lower or 'pivote' in trans_lower:
                return 'yawright'
            return 'right'
        
        # 3. Aucun mot-clé détecté
        return 'none'
    
    def _normalize_command(self, joystick_command: str) -> str:
        """Normalise les noms de commandes joystick"""
        if joystick_command == 'backward':
            return 'back'
        return joystick_command
    
    def process_file_pair(
        self, 
        main_tg_path: Path, 
        commands_tg_path: Path,
        audio_file: str
    ) -> List[AudioSegment]:
        """
        Traite une paire de fichiers TextGrid pour créer des segments audio labellisés
        
        Args:
            main_tg_path: Chemin vers le TextGrid principal
            commands_tg_path: Chemin vers le TextGrid _commands
            audio_file: Nom du fichier audio correspondant
            
        Returns:
            Liste d'AudioSegment prêts pour l'entraînement
        """
        # Parser les deux TextGrid
        transcription_segments = self.parse_main_textgrid(main_tg_path)
        joystick_commands = self.parse_commands_textgrid(commands_tg_path)
        
        audio_segments = []
        
        for seg in transcription_segments:
            # Trouver la commande joystick correspondante
            joy_cmd, confidence = self.find_matching_command(
                seg['start'], 
                seg['end'], 
                joystick_commands
            )
            
            # Classifier le segment
            command = self.classify_segment(seg['transcription'], joy_cmd)
            
            # Créer l'AudioSegment
            audio_seg = AudioSegment(
                audio_file=audio_file,
                start=seg['start'],
                end=seg['end'],
                duration=seg['end'] - seg['start'],
                transcription=seg['transcription'],
                command=command,
                joystick_match_confidence=confidence,
                distance_to_target=seg.get('distance')
            )
            
            audio_segments.append(audio_seg)
        
        return audio_segments
    
    def create_dataset(
        self,
        textgrid_dir: Path,
        output_csv: Path,
        audio_extension: str = '.wav'
    ) -> pd.DataFrame:
        """
        Crée un dataset complet à partir d'un répertoire de TextGrid
        
        Args:
            textgrid_dir: Répertoire contenant les fichiers TextGrid
            output_csv: Chemin de sortie pour le CSV
            audio_extension: Extension des fichiers audio
            
        Returns:
            DataFrame pandas avec toutes les informations
        """
        all_segments = []
        
        # Trouver toutes les paires de TextGrid (principal + _commands)
        main_textgrids = list(textgrid_dir.glob('*[!_commands].TextGrid'))
        
        print(f"Traitement de {len(main_textgrids)} fichiers...")
        
        for main_tg in main_textgrids:
            # Construire le chemin du fichier _commands correspondant
            base_name = main_tg.stem
            commands_tg = textgrid_dir / f"{base_name}_commands.TextGrid"
            
            if not commands_tg.exists():
                warnings.warn(f"Fichier _commands manquant pour {main_tg.name}")
                continue
            
            # Construire le nom du fichier audio
            audio_file = f"{base_name}{audio_extension}"
            
            # Traiter la paire de fichiers
            segments = self.process_file_pair(main_tg, commands_tg, audio_file)
            all_segments.extend(segments)
        
        # Convertir en DataFrame
        df = pd.DataFrame([
            {
                'audio_file': seg.audio_file,
                'start': seg.start,
                'end': seg.end,
                'duration': seg.duration,
                'transcription': seg.transcription,
                'command': seg.command,
                'joystick_confidence': seg.joystick_match_confidence,
                'distance_to_target': seg.distance_to_target
            }
            for seg in all_segments
        ])
        
        # Sauvegarder
        df.to_csv(output_csv, index=False, encoding='utf-8')
        
        print(f"\nDataset créé avec {len(df)} segments")
        print(f"Sauvegardé dans: {output_csv}")
        print("\nDistribution des classes:")
        print(df['command'].value_counts())
        print(f"\nConfiance moyenne du matching joystick: {df['joystick_confidence'].mean():.2f}")
        
        return df


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    textgrid_dir = Path("/path/to/textgrid/directory")
    output_csv = Path("/home/claude/voicestick_dataset.csv")
    
    # Créer le parser
    parser = VoiceStickParser(tolerance=1.5)
    
    # Créer le dataset complet
    df = parser.create_dataset(
        textgrid_dir=textgrid_dir,
        output_csv=output_csv,
        audio_extension='.wav'
    )
    
    # Afficher quelques exemples
    print("\n=== Exemples de segments ===")
    print(df.head(10))
    
    # Statistiques sur les ambiguïtés résolues
    gauche_droite = df[df['transcription'].str.contains('gauche|droite', case=False)]
    print(f"\n=== Résolution ambiguïté gauche/droite ===")
    print(f"Total segments gauche/droite: {len(gauche_droite)}")
    print(gauche_droite['command'].value_counts())
