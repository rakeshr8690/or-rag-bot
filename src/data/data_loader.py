"""
Data loader for downloading and loading OR datasets from HuggingFace.
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

from ..config.settings import DATA_RAW_PATH, DATASET_URLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ORDataLoader:
    """Loads Operations Research datasets from various sources."""
    
    def __init__(self, data_path: Path = DATA_RAW_PATH):
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def download_nl4opt(self) -> pd.DataFrame:
        """
        Download NL4Opt dataset from HuggingFace.
        
        Returns:
            DataFrame with problem descriptions and solutions
        """
        logger.info("Downloading NL4Opt dataset...")
        try:
            dataset = load_dataset(DATASET_URLS["nl4opt"])
            
            df = pd.DataFrame(dataset['train'])
            
            output_path = self.data_path / "nl4opt.parquet"
            df.to_parquet(output_path)
            logger.info(f"NL4Opt dataset saved to {output_path}")
            logger.info(f"Loaded {len(df)} problems")
            
            return df
        except Exception as e:
            logger.error(f"Error downloading NL4Opt: {e}")
            raise
            
    def download_industry_or(self) -> pd.DataFrame:
        """
        Download IndustryOR dataset from HuggingFace.
        
        Returns:
            DataFrame with industrial OR problems
        """
        logger.info("Downloading IndustryOR dataset...")
        try:
            dataset = load_dataset(DATASET_URLS["industry_or"])
            
            df = pd.DataFrame(dataset['train'])
            
            output_path = self.data_path / "industry_or.parquet"
            df.to_parquet(output_path)
            logger.info(f"IndustryOR dataset saved to {output_path}")
            logger.info(f"Loaded {len(df)} problems")
            
            return df
        except Exception as e:
            logger.error(f"Error downloading IndustryOR: {e}")
            raise
            
    def load_local_dataset(self, filename: str) -> pd.DataFrame:
        """
        Load a dataset from local storage.
        
        Args:
            filename: Name of the file (with extension)
            
        Returns:
            DataFrame with the dataset
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
            
        if filename.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif filename.endswith('.csv'):
            return pd.read_csv(file_path)
        elif filename.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
            
    def download_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Download all available datasets.
        
        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        datasets = {}
        
        try:
            datasets['nl4opt'] = self.download_nl4opt()
        except Exception as e:
            logger.warning(f"Failed to download NL4Opt: {e}")
            
        try:
            datasets['industry_or'] = self.download_industry_or()
        except Exception as e:
            logger.warning(f"Failed to download IndustryOR: {e}")
            
        logger.info(f"Successfully loaded {len(datasets)} datasets")
        return datasets
        
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about a dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_problems': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
        }
        
        if 'difficulty' in df.columns:
            stats['difficulty_distribution'] = df['difficulty'].value_counts().to_dict()
            
        if 'problem_type' in df.columns:
            stats['problem_type_distribution'] = df['problem_type'].value_counts().to_dict()
            
        return stats