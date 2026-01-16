"""
Data loader for 17Lands draft CSV files.

This module handles loading and preprocessing draft data from 17Lands
CSV files into DraftSequence objects for training.
"""

import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

from . import DraftSequence


logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


class DraftDataLoader:
    """
    Loads draft data from 17Lands CSV files.
    
    Processes game logs into individual pick sequences, filtering by
    player performance and extracting pool/pack/pick information.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Root directory containing set subdirectories
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise DataLoadError(f"Data directory not found: {data_dir}")
    
    def load_set_data(
        self,
        set_code: str,
        min_win_rate: float = 0.60,
        limit: Optional[int] = None
    ) -> List[DraftSequence]:
        """
        Load draft sequences from a set's CSV file.
        
        Args:
            set_code: MTG set code (e.g., 'MH3', 'BLB')
            min_win_rate: Minimum player win rate to include (default: 0.60)
            limit: Maximum number of rows to process (default: None for all)
        
        Returns:
            List of DraftSequence objects
        
        Raises:
            DataLoadError: If CSV file not found or processing fails
        """
        try:
            csv_path = self._find_csv_file(set_code)
            logger.info(f"Loading data from {csv_path}")
            
            # Read CSV (pandas handles .gz compression automatically)
            df = pd.read_csv(csv_path, nrows=limit)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            # Filter by win rate
            if 'user_game_win_rate_bucket' in df.columns:
                df = df[df['user_game_win_rate_bucket'] >= min_win_rate]
                logger.info(f"Filtered to {len(df)} rows with win rate >= {min_win_rate}")
            else:
                logger.warning("Column 'user_game_win_rate_bucket' not found, skipping win rate filter")
            
            # Extract draft sequences
            sequences = self._extract_sequences(df, set_code)
            logger.info(f"Extracted {len(sequences)} draft sequences")
            
            return sequences
            
        except Exception as e:
            raise DataLoadError(f"Failed to load data for set {set_code}: {e}") from e
    
    def load_multi_set_data(
        self,
        set_codes: List[str],
        **kwargs
    ) -> List[DraftSequence]:
        """
        Load data from multiple sets for general model training.
        
        Args:
            set_codes: List of MTG set codes (e.g., ['MH3', 'BLB'])
            **kwargs: Additional arguments passed to load_set_data()
        
        Returns:
            Combined list of DraftSequence objects from all sets
        
        Raises:
            DataLoadError: If any set fails to load
        """
        all_sequences = []
        
        for set_code in set_codes:
            try:
                sequences = self.load_set_data(set_code, **kwargs)
                all_sequences.extend(sequences)
                logger.info(f"Loaded {len(sequences)} sequences from {set_code}")
            except DataLoadError as e:
                logger.error(f"Failed to load {set_code}: {e}")
                raise
        
        logger.info(f"Total sequences loaded: {len(all_sequences)}")
        return all_sequences
    
    def _find_csv_file(self, set_code: str) -> Path:
        """
        Find the CSV file for a given set code.
        
        Args:
            set_code: MTG set code
        
        Returns:
            Path to CSV file
        
        Raises:
            DataLoadError: If CSV file not found
        """
        set_dir = self.data_dir / set_code
        
        if not set_dir.exists():
            raise DataLoadError(f"Set directory not found: {set_dir}")
        
        # Look for CSV or CSV.GZ files
        csv_files = list(set_dir.glob("*.csv")) + list(set_dir.glob("*.csv.gz"))
        
        if not csv_files:
            raise DataLoadError(f"No CSV files found in {set_dir}")
        
        if len(csv_files) > 1:
            logger.warning(f"Multiple CSV files found in {set_dir}, using first: {csv_files[0]}")
        
        return csv_files[0]
    
    def _extract_sequences(self, df: pd.DataFrame, set_code: str) -> List[DraftSequence]:
        """
        Extract DraftSequence objects from DataFrame.
        
        Args:
            df: DataFrame with draft data
            set_code: MTG set code for logging
        
        Returns:
            List of DraftSequence objects
        """
        sequences = []
        
        # Get column names for pack and pool
        pack_cols = [col for col in df.columns if col.startswith('pack_card_')]
        pool_cols = [col for col in df.columns if col.startswith('pool_')]
        
        if not pack_cols:
            raise DataLoadError("No pack_card_ columns found in CSV")
        
        if not pool_cols:
            raise DataLoadError("No pool_ columns found in CSV")
        
        # Process each row (each pick)
        for idx, row in df.iterrows():
            try:
                # Extract draft metadata
                draft_id = str(row.get('draft_id', f'{set_code}_{idx}'))
                pick_number = int(row.get('pick_number', 0))
                picked_card = str(row.get('pick', ''))
                
                if not picked_card or pd.isna(picked_card):
                    continue
                
                # Extract pack (cards with value 1 in pack_card_ columns)
                pack = []
                for col in pack_cols:
                    if row[col] == 1:
                        card_name = col.replace('pack_card_', '')
                        pack.append(card_name)
                
                if not pack:
                    logger.debug(f"Skipping row {idx}: empty pack")
                    continue
                
                # Extract pool (cards with value > 0 in pool_ columns)
                pool = []
                for col in pool_cols:
                    count = row[col]
                    if pd.notna(count) and count > 0:
                        card_name = col.replace('pool_', '')
                        # Add card multiple times if count > 1
                        for _ in range(int(count)):
                            pool.append(card_name)
                
                # Create sequence
                sequence = DraftSequence(
                    draft_id=draft_id,
                    pick_number=pick_number,
                    pool=pool,
                    pack=pack,
                    picked_card=picked_card
                )
                
                # Validate
                try:
                    sequence.validate()
                    sequences.append(sequence)
                except ValueError as e:
                    logger.debug(f"Skipping invalid sequence at row {idx}: {e}")
                    continue
                
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
        
        return sequences


__all__ = ['DraftDataLoader', 'DataLoadError']
