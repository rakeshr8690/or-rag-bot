"""
Data processor for cleaning and enriching OR problem datasets.
"""

import pandas as pd
import re
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ORDataProcessor:
    """Process and clean OR problem data."""
    
    def __init__(self):
        self.problem_types = ['LP', 'MIP', 'IP', 'NLP', 'MILP']
        self.industries = [
            'Manufacturing', 'Supply Chain', 'Logistics', 'Finance',
            'Healthcare', 'Transportation', 'Energy', 'Agriculture'
        ]
        
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing HTML tags, extra whitespace, and fixing encoding.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = re.sub(r'<[^>]+>', '', str(text))
        
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        text = ' '.join(text.split())
        
        return text
        
    def extract_problem_type(self, text: str) -> str:
        """
        Extract problem type from text.
        
        Args:
            text: Problem description
            
        Returns:
            Problem type (LP, MIP, etc.)
        """
        text_lower = text.lower()
        
        if 'integer' in text_lower or 'binary' in text_lower or '0-1' in text_lower:
            if 'mixed' in text_lower or ('integer' in text_lower and 'continuous' in text_lower):
                return 'MIP'
            return 'IP'
            
        if any(term in text_lower for term in ['nonlinear', 'quadratic', 'exponential', 'logarithmic']):
            return 'NLP'
            
        return 'LP'
        
    def infer_industry(self, text: str) -> str:
        """
        Infer industry from problem description.
        
        Args:
            text: Problem description
            
        Returns:
            Industry category
        """
        text_lower = text.lower()
        
        industry_keywords = {
            'Manufacturing': ['factory', 'production', 'machine', 'assembly', 'product'],
            'Supply Chain': ['supplier', 'distribution', 'warehouse', 'inventory', 'stock'],
            'Logistics': ['transportation', 'shipping', 'delivery', 'route', 'vehicle'],
            'Finance': ['investment', 'portfolio', 'capital', 'budget', 'profit', 'cost'],
            'Healthcare': ['patient', 'hospital', 'medical', 'treatment', 'doctor'],
            'Transportation': ['route', 'travel', 'distance', 'vehicle', 'freight'],
            'Energy': ['power', 'electricity', 'energy', 'grid', 'renewable'],
            'Agriculture': ['crop', 'farm', 'harvest', 'agriculture', 'field']
        }
        
        scores = {}
        for industry, keywords in industry_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[industry] = score
            
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return 'General'
        
    def estimate_difficulty(self, text: str, num_variables: Optional[int] = None, 
                           num_constraints: Optional[int] = None) -> str:
        """
        Estimate problem difficulty.
        
        Args:
            text: Problem description
            num_variables: Number of decision variables
            num_constraints: Number of constraints
            
        Returns:
            Difficulty level (Easy/Medium/Hard)
        """
        complexity_score = 0
        
        text_lower = text.lower()
        
        complex_terms = ['nonlinear', 'integer', 'binary', 'mixed', 'multiple objectives']
        complexity_score += sum(2 for term in complex_terms if term in text_lower)
        
        if num_variables:
            if num_variables > 20:
                complexity_score += 3
            elif num_variables > 10:
                complexity_score += 2
            elif num_variables > 5:
                complexity_score += 1
                
        if num_constraints:
            if num_constraints > 15:
                complexity_score += 3
            elif num_constraints > 8:
                complexity_score += 2
            elif num_constraints > 4:
                complexity_score += 1
        
        if complexity_score >= 8:
            return 'Hard'
        elif complexity_score >= 4:
            return 'Medium'
        else:
            return 'Easy'
            
    def extract_numerical_values(self, text: str) -> List[float]:
        """
        Extract numerical values from text.
        
        Args:
            text: Text containing numbers
            
        Returns:
            List of numerical values
        """
        numbers = re.findall(r'\d+\.?\d*', text)
        return [float(num) for num in numbers]
        
    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a complete dataset.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing {len(df)} problems...")
        
        processed_df = df.copy()
        
        text_columns = processed_df.select_dtypes(include=['object']).columns
        for col in tqdm(text_columns, desc="Cleaning text"):
            processed_df[col] = processed_df[col].apply(self.clean_text)
        
        if 'problem_type' not in processed_df.columns:
            logger.info("Inferring problem types...")
            processed_df['problem_type'] = processed_df.iloc[:, 0].apply(self.extract_problem_type)
            
        if 'industry' not in processed_df.columns:
            logger.info("Inferring industries...")
            processed_df['industry'] = processed_df.iloc[:, 0].apply(self.infer_industry)
            
        if 'difficulty' not in processed_df.columns:
            logger.info("Estimating difficulty...")
            processed_df['difficulty'] = processed_df.iloc[:, 0].apply(
                lambda x: self.estimate_difficulty(x)
            )
        
        if 'problem_id' not in processed_df.columns:
            processed_df['problem_id'] = [
                f"{row['problem_type']}_{row['industry'][:3].upper()}_{i:04d}"
                for i, row in processed_df.iterrows()
            ]
            
        logger.info("Processing complete!")
        return processed_df
        
    def deduplicate(self, df: pd.DataFrame, text_column: str = None) -> pd.DataFrame:
        """
        Remove duplicate problems.
        
        Args:
            df: DataFrame to deduplicate
            text_column: Column to check for duplicates
            
        Returns:
            Deduplicated DataFrame
        """
        if text_column:
            before_count = len(df)
            df = df.drop_duplicates(subset=[text_column], keep='first')
            after_count = len(df)
            logger.info(f"Removed {before_count - after_count} duplicates")
        
        return df
        
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate processed data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation report
        """
        report = {
            'total_problems': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'valid_problems': 0,
            'issues': []
        }
        
        required_columns = ['problem_id', 'problem_type', 'industry', 'difficulty']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            report['issues'].append(f"Missing columns: {missing_columns}")
        else:
            report['valid_problems'] = len(df.dropna(subset=required_columns))
            
        return report