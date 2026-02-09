"""
Parser simplifié pour annotations directes TextGrid
Spécialement adapté pour le format UTF-16 avec tier "commands"

Usage:
    python parser_simple_final.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import warnings


class SimpleCommandParser:
    """
    Parser pour TextGrid avec annotations directes de commandes
    Gère l'encodage UTF-16 avec BOM
    """
    
    # Classes de commandes valides
    VALID_COMMANDS = [
        'forward', 'backward', 'left', 'right', 
        'up', 'down', 'yawleft', 'yawright', 'none'
    ]
    
    def __init__(self):
        pass
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Détecte l'encodage du fichier (UTF-8 ou UTF-16)"""
        try:
            # Essayer de lire les premiers octets
            with open(file_path, 'rb') as f:
                first_bytes = f.read(4)
                
            # Vérifier BOM UTF-16
            if first_bytes[:2] == b'\xff\xfe' or first_bytes[:2] == b'\xfe\xff':
                return 'utf-16'
            else:
                return 'utf-8'
        except Exception:
            return 'utf-8'  # Par défaut
    
    def _parse_textgrid_manual(self, tg_path: Path) -> Dict:
        """
        Parse manuellement un fichier TextGrid
        Gère les encodages UTF-8 et UTF-16
        """
        # Détecter l'encodage
        encoding = self._detect_encoding(tg_path)
        
        try:
            with open(tg_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
        except Exception as e:
            # Si UTF-16 échoue, essayer UTF-8
            try:
                with open(tg_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception:
                raise ValueError(f"Impossible de lire {tg_path}: {e}")
        
        tiers = []
        current_tier = None
        current_interval = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Début d'un nouveau tier
            if 'class = "IntervalTier"' in line or 'class = " I n t e r v a l T i e r "' in line:
                if current_tier:
                    tiers.append(current_tier)
                current_tier = {'name': None, 'intervals': []}
            
            # Nom du tier
            elif 'name =' in line:
                # Extraire le nom (gérer les espaces UTF-16)
                parts = line.split('=')
                if len(parts) > 1:
                    name = parts[1].strip().strip('"').replace(' ', '')
                    if current_tier is not None:
                        current_tier['name'] = name.lower()
            
            # Début d'un interval (vérifier que ce n'est pas "intervals: size")
            elif line.startswith('xmin') and 'intervals:' not in ''.join(lines[max(0, i-5):i]):
                if current_interval and current_tier:
                    current_tier['intervals'].append(current_interval)
                
                try:
                    xmin_str = line.split('=')[1].strip()
                    xmin = float(xmin_str)
                    current_interval = {'xmin': xmin, 'xmax': None, 'text': ''}
                except (IndexError, ValueError):
                    current_interval = None
            
            # xmax de l'interval
            elif line.startswith('xmax') and current_interval is not None:
                try:
                    xmax_str = line.split('=')[1].strip()
                    xmax = float(xmax_str)
                    current_interval['xmax'] = xmax
                except (IndexError, ValueError):
                    pass
            
            # Texte de l'interval
            elif line.startswith('text') and current_interval is not None:
                parts = line.split('=', 1)
                if len(parts) > 1:
                    # Enlever les guillemets et les espaces UTF-16
                    text = parts[1].strip().strip('"').replace(' ', '')
                    current_interval['text'] = text
            
            i += 1
        
        # Ajouter le dernier interval et tier
        if current_interval and current_tier:
            current_tier['intervals'].append(current_interval)
        if current_tier:
            tiers.append(current_tier)
        
        return {'tiers': tiers}
    
    def parse_annotated_textgrid(self, tg_path: Path, tier_name: str = "commands") -> List[Dict]:
        """
        Parse un TextGrid avec annotations directes de commandes
        
        Args:
            tg_path: Chemin vers le fichier TextGrid
            tier_name: Nom du tier contenant les commandes (défaut: "commands")
        
        Returns:
            Liste de dictionnaires avec start, end, command
        """
        try:
            tg = self._parse_textgrid_manual(tg_path)
        except Exception as e:
            warnings.warn(f"Erreur lors du parsing de {tg_path}: {e}")
            return []
        
        # Trouver le tier de commandes (insensible à la casse)
        command_tier = None
        tier_name_lower = tier_name.lower()
        
        for tier in tg['tiers']:
            if tier['name'] and tier['name'].lower() == tier_name_lower:
                command_tier = tier
                break
        
        if not command_tier:
            available_tiers = [t['name'] for t in tg['tiers'] if t['name']]
            warnings.warn(f"Tier '{tier_name}' non trouvé dans {tg_path}. Tiers disponibles: {available_tiers}")
            return []
        
        segments = []
        for interval in command_tier['intervals']:
            command = interval['text'].strip().lower()
            
            # Ignorer les intervalles vides
            if not command:
                continue
            
            # Vérifier que la commande est valide
            if command not in self.VALID_COMMANDS:
                warnings.warn(f"Commande invalide '{command}' à {interval['xmin']:.2f}s dans {tg_path.name} - ignorée")
                continue
            
            segments.append({
                'start': interval['xmin'],
                'end': interval['xmax'],
                'duration': interval['xmax'] - interval['xmin'],
                'command': command
            })
        
        return segments
    
    def create_dataset_from_annotations(
        self,
        textgrid_dir: Path,
        audio_dir: Path,
        output_csv: Path,
        tier_name: str = "commands",
        audio_extension: str = '.wav'
    ) -> pd.DataFrame:
        """
        Crée un dataset complet à partir d'annotations directes
        
        Args:
            textgrid_dir: Répertoire contenant les fichiers TextGrid annotés
            audio_dir: Répertoire contenant les fichiers audio correspondants
            output_csv: Chemin de sortie pour le CSV
            tier_name: Nom du tier contenant les commandes
            audio_extension: Extension des fichiers audio
        
        Returns:
            DataFrame pandas avec toutes les informations
        """
        all_segments = []
        
        # Vérifier que le répertoire existe
        if not textgrid_dir.exists():
            raise FileNotFoundError(f"Le répertoire n'existe pas: {textgrid_dir}")
        
        # Afficher tous les fichiers présents
        print(f"📁 Contenu du répertoire {textgrid_dir}:")
        all_files = list(textgrid_dir.iterdir())
        for f in all_files[:10]:  # Afficher les 10 premiers
            print(f"  - {f.name}")
        if len(all_files) > 10:
            print(f"  ... et {len(all_files) - 10} autres fichiers")
        print()
        
        # Trouver tous les TextGrid (essayer plusieurs patterns)
        textgrids = list(textgrid_dir.glob('*.TextGrid'))
        if not textgrids:
            textgrids = list(textgrid_dir.glob('*.textgrid'))
        if not textgrids:
            textgrids = list(textgrid_dir.glob('*.TEXTGRID'))
        if not textgrids:
            # Essayer de trouver n'importe quel fichier contenant "TextGrid"
            textgrids = [f for f in textgrid_dir.iterdir() if 'textgrid' in f.name.lower()]
        
        if not textgrids:
            raise FileNotFoundError(
                f"Aucun fichier TextGrid trouvé dans {textgrid_dir}\n"
                f"Vérifiez que les fichiers ont l'extension .TextGrid\n"
                f"Fichiers trouvés: {[f.name for f in all_files[:5]]}"
            )
        
        print(f"Traitement de {len(textgrids)} fichier(s) TextGrid...")
        
        for tg_file in textgrids:
            # Nom du fichier audio correspondant
            audio_file = tg_file.stem + audio_extension
            audio_path = audio_dir / audio_file
            
            # Vérifier que l'audio existe
            if not audio_path.exists():
                warnings.warn(f"Fichier audio manquant: {audio_path}")
                # Continuer quand même pour créer le dataset
            
            # Parser le TextGrid
            segments = self.parse_annotated_textgrid(tg_file, tier_name)
            
            if not segments:
                warnings.warn(f"Aucun segment valide trouvé dans {tg_file.name}")
                continue
            
            # Ajouter le nom du fichier audio à chaque segment
            for seg in segments:
                seg['audio_file'] = audio_file
                all_segments.append(seg)
            
            print(f"  ✓ {tg_file.name}: {len(segments)} segments")
        
        if not all_segments:
            raise ValueError("Aucun segment trouvé dans les fichiers TextGrid")
        
        # Convertir en DataFrame
        df = pd.DataFrame(all_segments)
        
        # Réorganiser les colonnes
        df = df[['audio_file', 'start', 'end', 'duration', 'command']]
        
        # Sauvegarder
        output_csv.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        
        print(f"\n✓ Dataset créé avec {len(df)} segments")
        print(f"  Sauvegardé dans: {output_csv}")
        print(f"  Fichiers audio traités: {df['audio_file'].nunique()}")
        
        print("\n📊 Distribution des classes:")
        for cmd, count in df['command'].value_counts().items():
            pct = (count / len(df)) * 100
            print(f"  {cmd:12s}: {count:3d} segments ({pct:5.1f}%)")
        
        print(f"\n⏱️  Durée moyenne des segments: {df['duration'].mean():.2f}s")
        print(f"  Durée min: {df['duration'].min():.2f}s")
        print(f"  Durée max: {df['duration'].max():.2f}s")
        
        return df


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    # ⚠️ IMPORTANT: REMPLACEZ CES CHEMINS PAR VOS VRAIS CHEMINS
    
    TEXTGRID_DIR = Path(r"B:\drones\nikita_07-01-26\complete")
    AUDIO_DIR = Path(r"B:\drones\nikita_07-01-26\complete")
    OUTPUT_CSV = Path(r"B:\drones\nikita_07-01-26\dataset_commands.csv")
    
    print("=" * 80)
    print("PARSER TEXTGRID SIMPLIFIÉ - ANNOTATIONS DIRECTES")
    print("=" * 80)
    print()
    print(f"📁 Répertoire TextGrid: {TEXTGRID_DIR}")
    print(f"🎵 Répertoire Audio: {AUDIO_DIR}")
    print(f"💾 Fichier de sortie: {OUTPUT_CSV}")
    print()
    
    try:
        # Créer le parser
        parser = SimpleCommandParser()
        
        # Créer le dataset
        df = parser.create_dataset_from_annotations(
            textgrid_dir=TEXTGRID_DIR,
            audio_dir=AUDIO_DIR,
            output_csv=OUTPUT_CSV,
            tier_name="commands",  # Le nom du tier
            audio_extension='.wav'
        )
        
        print("\n" + "=" * 80)
        print("✓ SUCCÈS")
        print("=" * 80)
        print(f"\nVous pouvez maintenant utiliser {OUTPUT_CSV} pour l'entraînement")
        
        # Afficher un aperçu
        print("\nAperçu des données:")
        print(df.head(10).to_string())
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
