"""
Test script to verify the carpenter problem formulation fix.
"""
import sys
sys.path.insert(0, r'c:\Study\9th sem\or-rag-bot')

from src.models.llm_handler import LLMHandler
from src.config.prompts import MERGED_FORMULATION_PROMPT

carpenter_problem = """
A carpenter makes tables and chairs. 
- A table takes 6 hours to produce and sells for $30 profit.
- A chair takes 3 hours to produce and sells for $10 profit.
- The carpenter has 40 hours available per week.
- Customer demand requires at least 3 times as many chairs as tables (xC >= 3*xT).
- Tables take up 4 times as much storage space as chairs.
- Storage space is limited to at most 4 tables' worth of space per week: (xC/4) + xT <= 4.
- The carpenter wants to maximize total profit.

Formulate this as a linear program.
"""

print("=" * 80)
print("CARPENTER PROBLEM FORMULATION TEST")
print("=" * 80)
print(f"\nProblem:\n{carpenter_problem}")
print("\n" + "=" * 80)
print("EXPECTED OUTPUT:")
print("=" * 80)
print("""
Variables:
- xT = number of tables made per week
- xC = number of chairs made per week

Constraints:
1. total work time: 6xT + 3xC <= 40
2. customer demand: xC >= 3xT
3. storage space: (xC/4) + xT <= 4
4. all variables >= 0

Objective:
maximize 30xT + 10xC
""")

print("\n" + "=" * 80)
print("ACTUAL BOT OUTPUT:")
print("=" * 80)

try:
    handler = LLMHandler()
    
    prompt = MERGED_FORMULATION_PROMPT.format(
        retrieved_context="",
        user_query=carpenter_problem
    )
    
    response = handler.generate_response(prompt)
    print(response)
    
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    
    issues = []
    
    if "xT = 0" in response or "x_T = 0" in response or "x_t = 0" in response:
        issues.append("ISSUE: Terminal condition (xT=0) added - should NOT be there")
    
    if "S*xt + yt" in response or "S·x_t" in response:
        issues.append("ISSUE: Wrong storage constraint form - parameterized incorrectly")
    
    if "(xC/4) + xT <= 4" in response:
        issues.append("CORRECT: Storage constraint formulated correctly")
    elif "xC/4" in response or "chairs/4" in response:
        issues.append("GOOD: Using fractional chair space concept")
    
    if "Σ" in response and "ht*xt" in response:
        issues.append("ISSUE: Time-indexed summation used for single-period problem")
    
    if "6xT + 3xC <= 40" in response or "6*xT + 3*xC <= 40" in response:
        issues.append("CORRECT: Production time constraint uses actual numbers")
    
    if "xC >= 3xT" in response or "xC >= 3*xT" in response:
        issues.append("CORRECT: Demand constraint formulated correctly")
    
    if "30xT + 10xC" in response or "30*xT + 10*xC" in response:
        issues.append("CORRECT: Objective uses actual profit values")
    
    if len(issues) == 0:
        print("No specific issues detected")
    else:
        print("\n".join(issues))

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
