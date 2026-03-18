from typing import Dict

class OptimizationHandler:
    def format_formulation(self, problem_data: Dict) -> str:
        """
        Returns only the model-generated formulation from the problem description.
        No template, no repeated sections, no formatting issues.
        """
        return problem_data.get('description', 'No problem description provided.')
