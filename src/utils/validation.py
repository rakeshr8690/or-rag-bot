"""
Validation utilities for OR problems and solutions.
"""

import re
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ORValidator:
    """Validate OR problems and solutions."""
    
    def __init__(self):
        self.required_fields = [
            'problem_id', 'problem_type', 'industry', 'difficulty'
        ]
        
    def validate_problem_structure(self, problem: Dict) -> Tuple[bool, List[str]]:
        """
        Validate that a problem has all required fields.
        
        Args:
            problem: Problem dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for field in self.required_fields:
            if field not in problem:
                errors.append(f"Missing required field: {field}")
                
        if 'problem_type' in problem:
            valid_types = ['LP', 'IP', 'MIP', 'MILP', 'NLP', 'MINLP']
            if problem['problem_type'] not in valid_types:
                errors.append(f"Invalid problem type: {problem['problem_type']}")
                
        if 'difficulty' in problem:
            valid_difficulties = ['Easy', 'Medium', 'Hard']
            if problem['difficulty'] not in valid_difficulties:
                errors.append(f"Invalid difficulty: {problem['difficulty']}")
                
        return len(errors) == 0, errors
        
    def validate_mathematical_formulation(self, formulation: str) -> Tuple[bool, str]:
        """
        Validate mathematical formulation syntax.
        
        Args:
            formulation: Mathematical formulation string
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not formulation or len(formulation.strip()) == 0:
            return False, "Formulation is empty"
            
        has_objective = any(word in formulation.lower() 
                          for word in ['maximize', 'minimize', 'max', 'min'])
        if not has_objective:
            return False, "No objective function found"
            
        has_constraints = any(word in formulation.lower() 
                            for word in ['subject to', 'constraints', 's.t.', 'such that'])
        if not has_constraints:
            return False, "No constraints section found"
            
        return True, "Formulation appears valid"
        
    def validate_code_syntax(self, code: str) -> Tuple[bool, str]:
        """
        Basic validation of Python code syntax.
        
        Args:
            code: Python code string
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            compile(code, '<string>', 'exec')
            return True, "Code syntax is valid"
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
            
    def validate_solution(self, solution: Dict) -> Tuple[bool, List[str]]:
        """
        Validate solution structure.
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if 'status' not in solution:
            errors.append("Solution missing status field")
        elif solution['status'] not in ['optimal', 'feasible', 'infeasible', 'unbounded', 'error']:
            errors.append(f"Invalid status: {solution['status']}")
            
        if solution.get('status') in ['optimal', 'feasible']:
            if 'objective_value' not in solution:
                errors.append("Missing objective value for feasible/optimal solution")
                
        # Check for variables
        if 'variables' not in solution:
            errors.append("Solution missing variables field")
            
        return len(errors) == 0, errors
        
    def validate_constraints(self, constraints: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate constraint definitions.
        
        Args:
            constraints: List of constraint strings
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not constraints:
            errors.append("No constraints provided")
            return False, errors
            
        for i, constraint in enumerate(constraints):
            # Check for comparison operators
            has_operator = any(op in constraint for op in ['<=', '>=', '=', '<', '>'])
            if not has_operator:
                errors.append(f"Constraint {i+1} missing comparison operator")
                
        return len(errors) == 0, errors
        
    def validate_dataset(self, dataset: List[Dict]) -> Dict:
        """
        Validate entire dataset.
        
        Args:
            dataset: List of problem dictionaries
            
        Returns:
            Validation report dictionary
        """
        report = {
            'total_problems': len(dataset),
            'valid_problems': 0,
            'invalid_problems': 0,
            'errors_by_type': {},
            'problem_issues': []
        }
        
        for idx, problem in enumerate(dataset):
            is_valid, errors = self.validate_problem_structure(problem)
            
            if is_valid:
                report['valid_problems'] += 1
            else:
                report['invalid_problems'] += 1
                report['problem_issues'].append({
                    'problem_id': problem.get('problem_id', f'index_{idx}'),
                    'errors': errors
                })
                
                # Count error types
                for error in errors:
                    error_type = error.split(':')[0]
                    report['errors_by_type'][error_type] = \
                        report['errors_by_type'].get(error_type, 0) + 1
                        
        report['validity_rate'] = report['valid_problems'] / report['total_problems'] \
                                 if report['total_problems'] > 0 else 0
                                 
        return report
        
    def check_data_quality(self, text: str) -> Dict:
        """
        Check quality metrics for text data.
        
        Args:
            text: Text to check
            
        Returns:
            Quality metrics dictionary
        """
        metrics = {
            'length': len(text),
            'word_count': len(text.split()),
            'has_numbers': bool(re.search(r'\d', text)),
            'has_equations': bool(re.search(r'[=<>]', text)),
            'has_units': bool(re.search(r'\b(units?|kg|tons?|hours?|dollars?)\b', text, re.IGNORECASE)),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_sentence_length': 0
        }
        
        if metrics['sentence_count'] > 0:
            metrics['avg_sentence_length'] = metrics['word_count'] / metrics['sentence_count']
            
        # Quality score (0-1)
        score = 0.0
        if metrics['length'] > 100:
            score += 0.2
        if metrics['word_count'] > 20:
            score += 0.2
        if metrics['has_numbers']:
            score += 0.2
        if metrics['has_equations']:
            score += 0.2
        if metrics['has_units']:
            score += 0.2
            
        metrics['quality_score'] = score
        
        return metrics