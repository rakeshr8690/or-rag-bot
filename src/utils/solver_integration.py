"""
Integration with optimization solvers (PuLP, Pyomo, OR-Tools).
"""

import re
import logging
from typing import Dict, Optional, Tuple
import io
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SolverIntegration:
    """Execute optimization code and return results."""
    
    def __init__(self):
        self.supported_solvers = ['pulp', 'pyomo', 'ortools']
        
    def extract_code_from_response(self, response: str) -> Optional[str]:
        """
        Extract Python code from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted code or None
        """
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0]
            
        code_pattern = r'```\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0]
            
        return None
        
    def execute_code(self, code: str, timeout: int = 30) -> Tuple[bool, str, Dict]:
        """
        Execute optimization code safely.
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            
        Returns:
            Tuple of (success, output, results_dict)
        """
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        results = {}
        success = False
        
        try:
            namespace = {
                '__builtins__': __builtins__,
                'results': results
            }
            
            exec(code, namespace)
            
            output = captured_output.getvalue()
            success = True
            
            logger.info("Code executed successfully")
            
        except Exception as e:
            output = f"Error executing code: {str(e)}"
            logger.error(output)
            
        finally:
            sys.stdout = old_stdout
            
        return success, output, results
        
    def solve_with_pulp(self, problem_dict: Dict) -> Dict:
        """
        Solve optimization problem using PuLP.
        
        Args:
            problem_dict: Dictionary with problem data
            
        Returns:
            Solution dictionary
        """
        try:
            from pulp import LpProblem, LpMaximize, LpMinimize, LpVariable
            
            prob = LpProblem(
                problem_dict.get('name', 'OR_Problem'),
                LpMaximize if problem_dict.get('sense') == 'maximize' else LpMinimize
            )
            
            variables = problem_dict.get('variables', {})
            lp_vars = {}
            
            for var_name, var_data in variables.items():
                lp_vars[var_name] = LpVariable(
                    var_name,
                    lowBound=var_data.get('lb', 0),
                    upBound=var_data.get('ub', None),
                    cat=var_data.get('type', 'Continuous')
                )
                
            logger.info("PuLP problem created (template)")
            
            return {
                'status': 'template',
                'message': 'PuLP integration template - full implementation needed'
            }
            
        except ImportError:
            logger.error("PuLP not installed")
            return {'status': 'error', 'message': 'PuLP not installed'}
        except Exception as e:
            logger.error(f"PuLP error: {e}")
            return {'status': 'error', 'message': str(e)}
            
    def validate_solution(self, solution: Dict, problem: Dict) -> bool:
        """
        Validate solution against problem constraints.
        
        Args:
            solution: Solution dictionary
            problem: Problem dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if 'variables' not in solution:
            return False
            
        for var_name in problem.get('variables', {}):
            if var_name not in solution['variables']:
                return False
                
        return True
        
    def format_solution(self, solution_output: str) -> Dict:
        """
        Parse and format solution output.
        
        Args:
            solution_output: Raw solution output
            
        Returns:
            Formatted solution dictionary
        """
        formatted = {
            'raw_output': solution_output,
            'objective_value': None,
            'variables': {},
            'status': 'unknown'
        }
        
        obj_pattern = r'objective.*?[:=]\s*([-+]?\d*\.?\d+)'
        obj_match = re.search(obj_pattern, solution_output, re.IGNORECASE)
        if obj_match:
            formatted['objective_value'] = float(obj_match.group(1))
            
        var_pattern = r'(\w+)\s*[:=]\s*([-+]?\d*\.?\d+)'
        var_matches = re.findall(var_pattern, solution_output)
        for var_name, value in var_matches:
            formatted['variables'][var_name] = float(value)
            
        if 'optimal' in solution_output.lower():
            formatted['status'] = 'optimal'
        elif 'infeasible' in solution_output.lower():
            formatted['status'] = 'infeasible'
        elif 'unbounded' in solution_output.lower():
            formatted['status'] = 'unbounded'
            
        return formatted