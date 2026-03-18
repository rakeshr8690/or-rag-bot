"""
Document structurer for formatting OR problems into standardized templates.
"""

import pandas as pd
from typing import Dict, List
import logging
from tqdm import tqdm
import json

from ..config.settings import PROBLEM_TEMPLATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentStructurer:
    """Structure OR problems into standardized document format."""
    
    def __init__(self):
        self.template = PROBLEM_TEMPLATE
        
    def structure_problem(self, problem_data: Dict) -> str:
        """
        Convert raw problem data into structured document.
        
        Args:
            problem_data: Dictionary containing problem information
            
        Returns:
            Formatted document string
        """
        structured_doc = self.template.format(
            problem_id=problem_data.get('problem_id', 'UNKNOWN'),
            problem_type=problem_data.get('problem_type', 'LP'),
            industry=problem_data.get('industry', 'General'),
            difficulty=problem_data.get('difficulty', 'Medium'),
            title=self._extract_title(problem_data),
            context=self._extract_context(problem_data),
            variables=self._extract_variables(problem_data),
            objective=self._extract_objective(problem_data),
            constraints=self._extract_constraints(problem_data),
            numerical_data=self._extract_numerical_data(problem_data),
            formulation=self._extract_formulation(problem_data),
            solution_approach=self._extract_solution_approach(problem_data),
            optimal_solution=problem_data.get('answer', 'Not available'),
            sensitivity_notes=self._generate_sensitivity_notes(problem_data),
            related_problems=problem_data.get('related_problems', 'None'),
            keywords=self._generate_keywords(problem_data)
        )
        
        return structured_doc
        
    def _extract_title(self, data: Dict) -> str:
        """Extract or generate problem title."""
        if 'title' in data:
            return data['title']
        
        text = data.get('question', data.get('description', ''))
        words = text.split()[:10]
        return ' '.join(words) + ('...' if len(text.split()) > 10 else '')
        
    def _extract_context(self, data: Dict) -> str:
        """Extract business context from problem description."""
        text = data.get('question', data.get('description', ''))
        
        sentences = text.split('.')
        context_sentences = sentences[:min(3, len(sentences))]
        
        return '. '.join(context_sentences) + '.'
        
    def _extract_variables(self, data: Dict) -> str:
        """Extract or infer decision variables."""
        text = data.get('question', data.get('description', '')).lower()
        
        variables = []
        
        if 'produce' in text or 'production' in text:
            variables.append("- x_i: Quantity of product i to produce (units)")
        if 'worker' in text or 'employee' in text:
            variables.append("- w_t: Number of workers assigned in period t (people)")
        if 'transport' in text or 'ship' in text:
            variables.append("- t_ij: Amount transported from location i to j (units)")
        if 'invest' in text:
            variables.append("- i_j: Investment amount in option j (dollars)")
            
        if not variables:
            variables.append("- x: Decision variable (to be determined)")
            
        return '\n'.join(variables)
        
    def _extract_objective(self, data: Dict) -> str:
        """Extract objective function."""
        text = data.get('question', data.get('description', '')).lower()
        
        if 'maximize' in text or 'maximum' in text or 'max' in text:
            if 'profit' in text:
                return "Maximize: Total profit"
            elif 'revenue' in text:
                return "Maximize: Total revenue"
            else:
                return "Maximize: Objective function value"
        elif 'minimize' in text or 'minimum' in text or 'min' in text:
            if 'cost' in text:
                return "Minimize: Total cost"
            elif 'time' in text:
                return "Minimize: Total time"
            else:
                return "Minimize: Objective function value"
        else:
            return "Objective: To be determined from problem context"
            
    def _extract_constraints(self, data: Dict) -> str:
        """Extract constraints from problem description."""
        text = data.get('question', data.get('description', ''))
        
        constraints = []
        constraint_num = 1
        
        constraint_keywords = [
            ('capacity', 'Capacity constraint'),
            ('demand', 'Demand requirement'),
            ('budget', 'Budget constraint'),
            ('time', 'Time constraint'),
            ('resource', 'Resource availability'),
            ('worker', 'Worker availability'),
            ('at least', 'Minimum requirement'),
            ('at most', 'Maximum limit'),
            ('no more than', 'Upper bound'),
            ('no less than', 'Lower bound')
        ]
        
        for keyword, constraint_type in constraint_keywords:
            if keyword in text.lower():
                constraints.append(f"{constraint_num}. {constraint_type}")
                constraint_num += 1
                
        if not constraints:
            constraints.append("1. Constraints to be determined from problem context")
            
        return '\n'.join(constraints)
        
    def _extract_numerical_data(self, data: Dict) -> str:
        """Extract numerical parameters."""
        import re
        
        text = data.get('question', data.get('description', ''))
        
        number_pattern = r'(\d+(?:\.\d+)?)\s*(\w+)?'
        matches = re.findall(number_pattern, text)
        
        if matches:
            data_lines = []
            for i, (num, unit) in enumerate(matches[:10], 1):
                data_lines.append(f"Parameter {i}: {num} {unit}")
            return '\n'.join(data_lines)
        
        return "Numerical data embedded in problem description"
        
    def _extract_formulation(self, data: Dict) -> str:
        """Extract or indicate mathematical formulation."""
        if 'formulation' in data:
            return data['formulation']
            
        return "Mathematical formulation to be derived from problem constraints and objective"
        
    def _extract_solution_approach(self, data: Dict) -> str:
        """Suggest solution approach based on problem type."""
        problem_type = data.get('problem_type', 'LP')
        
        approaches = {
            'LP': 'Simplex method or Interior Point method using PuLP/Pyomo with CBC or GLPK solver',
            'IP': 'Branch and Bound algorithm using PuLP with CBC solver',
            'MIP': 'Branch and Cut algorithm using Pyomo with CPLEX or Gurobi solver',
            'NLP': 'Nonlinear solver such as IPOPT using Pyomo'
        }
        
        return approaches.get(problem_type, 'Standard optimization solver')
        
    def _generate_sensitivity_notes(self, data: Dict) -> str:
        """Generate sensitivity analysis notes."""
        problem_type = data.get('problem_type', 'LP')
        industry = data.get('industry', 'General')
        
        notes = [
            f"- {problem_type} problems: Analyze shadow prices for binding constraints",
            f"- {industry} context: Focus on operational parameter sensitivity",
            "- Consider constraint relaxation scenarios",
            "- Evaluate right-hand side ranging for key constraints"
        ]
        
        return '\n'.join(notes)
        
    def _generate_keywords(self, data: Dict) -> str:
        """Generate searchable keywords."""
        keywords = set()
        
        keywords.add(data.get('problem_type', 'LP'))
        keywords.add(data.get('industry', 'General'))
        keywords.add(data.get('difficulty', 'Medium'))
        
        text = data.get('question', data.get('description', '')).lower()
        
        keyword_list = [
            'optimization', 'linear programming', 'integer programming',
            'production', 'scheduling', 'allocation', 'transportation',
            'capacity', 'constraint', 'maximize', 'minimize'
        ]
        
        for kw in keyword_list:
            if kw in text:
                keywords.add(kw)
                
        return ', '.join(sorted(keywords))
        
    def structure_dataset(self, df: pd.DataFrame) -> List[Dict]:
        """
        Structure entire dataset into documents.
        
        Args:
            df: DataFrame with problem data
            
        Returns:
            List of structured documents with metadata
        """
        logger.info(f"Structuring {len(df)} problems...")
        
        structured_docs = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Structuring"):
            problem_data = row.to_dict()
            
            structured_text = self.structure_problem(problem_data)
            
            doc = {
                'id': problem_data.get('problem_id', f'DOC_{idx}'),
                'text': structured_text,
                'metadata': {
                    'problem_id': problem_data.get('problem_id', f'DOC_{idx}'),
                    'problem_type': problem_data.get('problem_type', 'LP'),
                    'industry': problem_data.get('industry', 'General'),
                    'difficulty': problem_data.get('difficulty', 'Medium'),
                    'source_dataset': problem_data.get('source', 'unknown')
                }
            }
            
            structured_docs.append(doc)
            
        logger.info(f"Structured {len(structured_docs)} documents")
        return structured_docs
        
    def save_structured_docs(self, docs: List[Dict], output_path: str):
        """
        Save structured documents to file.
        
        Args:
            docs: List of structured documents
            output_path: Path to save file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved {len(docs)} documents to {output_path}")