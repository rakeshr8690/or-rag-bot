"""
Post-processing utilities for formatting LLM responses.
Handles LaTeX formatting, line breaks, and MathJax compatibility.
"""

import re
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Format LLM responses for proper display."""
    
    def __init__(self):
        self.latex_block_pattern = re.compile(r'\$\$[\s\S]*?\$\$', re.MULTILINE)
        self.inline_math_pattern = re.compile(r'\$[^\$]+\$')
        
    def format_response(self, response: str) -> str:
        """
        Format response to ensure proper LaTeX formatting and line breaks.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Formatted response with proper LaTeX and line breaks
        """
        response = self._remove_latex_environments(response)
        
        response = self._ensure_latex_blocks(response)
        
        response = self._format_latex_blocks(response)
        
        response = self._convert_plain_math_to_latex(response)
        
        response = self._fix_spacing(response)
        
        return response
    
    def _remove_latex_environments(self, text: str) -> str:
        """
        Remove LaTeX document environments (itemize, enumerate, etc.) that MathJax can't render.
        Convert them to plain text lists.
        """
        import re
        
        def replace_itemize(match):
            content = match.group(1)
            items = re.findall(r'\\item\s+(.+?)(?=\\item|$)', content, re.DOTALL)
            result = '\n'.join([f'- {item.strip()}' for item in items if item.strip()])
            return result
        
        def replace_enumerate(match):
            content = match.group(1)
            items = re.findall(r'\\item\s+(.+?)(?=\\item|$)', content, re.DOTALL)
            result = '\n'.join([f'{i+1}. {item.strip()}' for i, item in enumerate(items) if item.strip()])
            return result
        
        text = re.sub(r'\\begin\{itemize\}(.*?)\\end\{itemize\}', replace_itemize, text, flags=re.DOTALL)
        text = re.sub(r'\\begin\{itemize\}(.*?)\\end\{itemize\}', replace_itemize, text, flags=re.DOTALL)
        
        text = re.sub(r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}', replace_enumerate, text, flags=re.DOTALL)
        text = re.sub(r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}', replace_enumerate, text, flags=re.DOTALL)
        
        text = re.sub(r'\\item\s+', '- ', text)
        text = re.sub(r'\\item\s*', '- ', text)
        
        for env in ['itemize', 'enumerate', 'description', 'list']:
            text = re.sub(rf'\\begin\{{{env}\}}', '', text, flags=re.IGNORECASE)
            text = re.sub(rf'\\end\{{{env}\}}', '', text, flags=re.IGNORECASE)
        
        math_environments = ['aligned', 'align', 'equation', 'eqnarray', 'matrix', 'cases']
        def remove_non_math_env(match):
            env_name = match.group(1)
            if env_name.lower() not in math_environments:
                return match.group(2)
            return match.group(0)
        
        text = re.sub(r'\\begin\{([^}]+)\}(.*?)\\end\{\1\}', remove_non_math_env, text, flags=re.DOTALL)
        
        return text
    
    def _ensure_latex_blocks(self, text: str) -> str:
        """Ensure mathematical expressions are in LaTeX blocks."""
        has_latex = bool(self.latex_block_pattern.search(text))
        
        if not has_latex:
            math_patterns = [
                (r'Minimize\s+([Σ∑]\s*[^\n]+)', r'Minimize $$\\sum_{t=1}^{T} ...$$'),
                (r'Subject to\s+([^\n]+)', r'Subject to constraints:'),
            ]
            
            if re.search(r'[Σ∑∑]|\\sum|\\prod|X_|I_|D_|P_|S_', text):
                logger.warning("Found mathematical notation but no LaTeX blocks. Model may need better prompting.")
        
        return text
    
    def _format_latex_blocks(self, text: str) -> str:
        """
        Format LaTeX blocks with proper line breaks.
        Ensures multi-line formulations are readable.
        """
        def format_block(match):
            block = match.group(0)
            content = block[2:-2].strip()
            
            if '\\begin{aligned}' in content or '\\begin{align}' in content:
                content = re.sub(r'\s+', ' ', content)
                content = re.sub(r'\\\\\s*', r'\\\\\n        ', content)
                return f"$$\n{content}\n$$"
            
            if '\\text{Subject to}' in content or '\\text{Minimize}' in content:
                parts = []
                
                obj_match = re.search(r'\\text\{Minimize\}[^\\]*', content)
                if obj_match:
                    parts.append(obj_match.group(0))
                    content = content.replace(obj_match.group(0), '')
                
                constraint_pattern = r'[^\\]+(?:I_\{[^\}]+\}|X_\{[^\}]+\}|D_\{[^\}]+\})[^\\]*'
                constraints = re.findall(constraint_pattern, content)
                
                if constraints:
                    formatted = '\\begin{aligned}\n'
                    if parts:
                        formatted += '  ' + parts[0] + ' \\\\\n'
                    formatted += '  \\text{Subject to} \\quad & ' + ' \\\\\n  & '.join(constraints) + '\n\\end{aligned}'
                    return f"$$\n{formatted}\n$$"
            
            return f"$$\n{content}\n$$"
        
        return self.latex_block_pattern.sub(format_block, text)
    
    def _convert_plain_math_to_latex(self, text: str) -> str:
        """
        Convert plain text mathematical notation to LaTeX.
        This is a fallback for when the model outputs plain text.
        """
        conversions = [
            (r'Σ\s*\(([^)]+)\)', r'\\sum_{\1}'),
            (r'Σ\s*([a-zA-Z])_([a-zA-Z])', r'\\sum_{\2} \1_{\2}'),
            (r'([A-Z])_([a-z0-9]+)(?![_^])', r'\1_{\2}'),
            (r'<=\s*', r'\\le '),
            (r'>=\s*', r'\\ge '),
            (r'=\s*', r'= '),
            (r'\\forall\s*', r'\\forall '),
        ]
        
        def convert_in_text(match):
            content = match.group(0)
            if '$' in content or '\\' in content:
                return content
            
            for pattern, replacement in conversions:
                content = re.sub(pattern, replacement, content)
            return content
        
        result = []
        last_end = 0
        for match in self.latex_block_pattern.finditer(text):
            before = text[last_end:match.start()]
            result.append(before)
            result.append(match.group(0))
            last_end = match.end()
        
        if last_end < len(text):
            remaining = text[last_end:]
            result.append(remaining)
        
        return ''.join(result)
    
    def _fix_spacing(self, text: str) -> str:
        """Fix spacing around LaTeX blocks for better readability."""
        text = re.sub(r'\n(\$\$)', r'\n\n\1', text)
        text = re.sub(r'(\$\$)\n', r'\1\n\n', text)
        
        text = re.sub(r'\n{3,}', r'\n\n', text)
        
        return text.strip()
    
    def validate_formulation(self, response: str) -> Tuple[bool, list]:
        """
        Validate that the formulation is correct.
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        if not self.latex_block_pattern.search(response):
            warnings.append("No LaTeX blocks found - formulation may not render properly")
        
        if 'production' in response.lower() or 'inventory' in response.lower():
            response_no_spaces = response.replace(' ', '')
            
            uses_x_t = ('X_t' in response or 'X_{t}' in response) and 'p_t' not in response
            uses_p_t = 'p_t' in response or 'p_{t}' in response
            
            if uses_x_t and not uses_p_t:
                warnings.append("WARNING: Using X_t for production. Should use p_t for production quantity in period t.")
            
            has_inventory_balance_p = (
                'I_{t-1}' in response and 'p_t' in response and 'd_t' in response
            ) or (
                'I_{t-1}+p_t-d_t=I_t' in response_no_spaces or
                'I_{t-1} + p_t - d_t = I_t' in response
            )
            
            has_inventory_balance_x = (
                'I_{t-1}' in response and 'X_t' in response and 'D_t' in response
            ) or (
                'I_{t-1}+X_t-D_t=I_t' in response_no_spaces or
                'I_{t-1} + X_t - D_t = I_t' in response
            )
            
            has_inventory_balance = has_inventory_balance_p or has_inventory_balance_x
            
            has_terminal_constraint = (
                'I_6 = 0' in response or
                'I_T = 0' in response or
                'I_T=0' in response_no_spaces or
                re.search(r'I_\d+\s*=\s*0', response)
            )
            
            has_problematic_constraint = (
                'p_t >= d_t' in response or 
                'p_t ≥ d_t' in response or
                'p_t>=d_t' in response_no_spaces or
                'X_t >= D_t' in response or 
                'X_t ≥ D_t' in response or
                'X_t>=D_t' in response_no_spaces
            )
            
            if has_problematic_constraint and not has_inventory_balance:
                warnings.append("WARNING: Found p_t >= d_t (or X_t >= D_t) constraint in production planning problem. "
                              "This prevents using inventory. Consider inventory balance equations instead.")
            
            if 'I_t' in response and not has_inventory_balance:
                warnings.append("WARNING: Inventory variables found but no inventory balance equations detected. "
                              "Production planning with inventory requires: I_{t-1} + p_t - d_t = I_t")
            
            if 'I_t' in response and not has_terminal_constraint:
                if re.search(r'\b(month|period|time|horizon)\b', response, re.IGNORECASE):
                    warnings.append("WARNING: Terminal inventory constraint (I_T = 0) may be missing. "
                                  "If the problem specifies no leftover inventory at the end, include I_T = 0.")
        
        if not re.search(r'minimize|maximize|min|max', response, re.IGNORECASE):
            warnings.append("No objective function (minimize/maximize) found")
        
        if not re.search(r'subject to|constraints|s\.t\.|such that', response, re.IGNORECASE):
            warnings.append("No constraints section found")
        
        if 'max\{0' in response or 'max(0' in response or '\\max\{0' in response:
            if 'production' in response.lower() or 'inventory' in response.lower():
                warnings.append("WARNING: Found max{0, ...} function in formulation. "
                              "For production planning with inventory, use inventory balance equations "
                              "I_{t-1} + p_t - d_t = I_t instead of max functions.")
        
        if 'production' in response.lower() and 'inventory' in response.lower():
            obj_pattern = re.search(r'minimize|minimize', response, re.IGNORECASE)
            if obj_pattern:
                has_prod_cost_term = bool(re.search(r'c.*p.*p_t|P.*X_t|production.*cost', response, re.IGNORECASE))
                has_holding_cost_term = bool(re.search(r'h.*I_t|S.*I_t|holding.*I_t|8.*I_t', response, re.IGNORECASE))
                
                has_max_function = bool(re.search(r'max\{0|max\(0|\\max\{0', response, re.IGNORECASE))
                
                if has_max_function:
                    warnings.append("WARNING: Objective function uses max{0, ...} which is incorrect. "
                                  "For production planning, use inventory variables I_t and inventory balance equations.")
                elif not (has_prod_cost_term and has_holding_cost_term):
                    warnings.append("WARNING: Objective function may not have correct structure. "
                                  "Should have: sum of (production_cost × production) + holding_cost × sum of inventory")
        
        return len(warnings) == 0, warnings

