"""
Test script to verify bug fixes for NFL GM Agent.
"""
import logging
from tools import evaluate_player, simulate_team_impact
from gm_agent import GMReActAgent

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("="*80)
print("TESTING BUG FIXES")
print("="*80)

# Test 1: OL Position Misclassification
print("\n[TEST 1] OL Misclassification - Tristan Wirfs as TE instead of OL")
print("-" * 80)

result_no_hint = evaluate_player("Tristan Wirfs")
result_with_hint = evaluate_player("Tristan Wirfs", "ol")

print(f"Without position hint: {result_no_hint.get('position', '??')} - {result_no_hint.get('error', 'OK')}")
print(f"With position hint:    {result_with_hint.get('position', '??')} - {result_with_hint.get('error', 'OK')}")

# OL positions expand to t/g/c, so any of those is valid
if result_with_hint.get('position') in ['t', 'g', 'c'] and 'error' not in result_with_hint:
    print("✓ PASS: OL lookup with position hint works (found as T/G/C)")
else:
    print("✗ FAIL: OL lookup still broken")

# Test 2: Nick Bosa ED vs DI Classification
print("\n[TEST 2] Nick Bosa - ED vs DI Classification")
print("-" * 80)

result_auto = evaluate_player("Nick Bosa")
result_ed = evaluate_player("Nick Bosa", "ed")
result_di = evaluate_player("Nick Bosa", "di")

print(f"Auto-detect:  {result_auto.get('position', '??')} - Grade: {result_auto.get('predicted_grade')}")
print(f"ED hint:      {result_ed.get('position', '??')} - Grade: {result_ed.get('predicted_grade')}")
print(f"DI hint:      {result_di.get('position', '??')} - Grade: {result_di.get('predicted_grade')}")

if result_ed.get('position') == 'ed' and 'error' not in result_ed:
    print("✓ PASS: Nick Bosa correctly identified as ED with hint")
else:
    print("✗ FAIL: ED classification still broken")

# Test 3: NaN Grade Filtering
print("\n[TEST 3] NaN Grade Filtering - Mike Boone (RB)")
print("-" * 80)

result_boone = evaluate_player("Mike Boone", "rb")
grade = result_boone.get('predicted_grade')

print(f"Mike Boone predicted_grade: {grade}")
if grade is None or (isinstance(grade, float) and str(grade) == 'nan'):
    print("✗ FAIL: NaN grade not filtered (still NaN)")
elif isinstance(grade, (int, float)) and not str(grade) == 'NaN':
    print(f"✓ PASS: NaN filtered, using fallback grade: {grade}")
else:
    print(f"? UNCLEAR: Grade is {grade} (type: {type(grade)})")

# Test 4: Silent FileNotFoundError Handling
print("\n[TEST 4] CSV Missing Error Logging")
print("-" * 80)
print("Check logs above for 'FileNotFoundError' or 'CSV file missing' messages")
print("(Would appear if any position CSV is missing)")

# Test 5: Impact Calculation with NaN Check
print("\n[TEST 5] Impact Calculation - NaN Grade Rejection")
print("-" * 80)

# Mock a move with NaN grade
nan_move = {
    "team": "SEA",
    "position": "rb",
    "player_grade": float('nan'),
    "season": 2023
}

result = simulate_team_impact(nan_move)
if "error" in result:
    print(f"✓ PASS: NaN grade rejected - {result['error']}")
else:
    print(f"✗ FAIL: NaN grade not rejected - {result}")

# Test 6: Best Candidate Selection
print("\n[TEST 6] Best Candidate Selection by Win Impact")
print("-" * 80)

agent = GMReActAgent()
# Create a simple test with safety candidates (should have measurable impact)
candidates = [
    ("Jamal Adams", "s"),
    ("Kevin Byard", "s"),
    ("Andrew Adams", "s"),
]

decision = agent.run("SEA", candidates, season=2023)

print(f"Candidates tested: {len(decision.get('Observation', []))} candidates evaluated")
final = decision.get('Final Decision', 'UNKNOWN')
impact = decision.get('Expected Win Impact', 0)

print(f"Final Decision: {final}")
print(f"Expected Win Impact: {impact} games")

if impact > 0 and "SIGN" in final:
    print("✓ PASS: Best candidate selected with positive impact")
elif impact <= 0 or "PASS" in final:
    print("⚠️  NO POSITIVE IMPACT: Agent correctly passed (no candidate improved team)")
else:
    print(f"? UNCLEAR: Unexpected result")

# Test 7: Position-Specific Weakness Resolution
print("\n[TEST 7] Weakness Resolution - OL and ED")
print("-" * 80)

# Get team context
from tools import get_team_context
team_ctx = get_team_context("SEA", 2023)
weak_pos = team_ctx.get('positional_grades', {})

print(f"SEA Positional Grades:")
for pos, grade in sorted(weak_pos.items(), key=lambda x: x[1] if x[1] else 100):
    print(f"  {pos.upper()}: {grade}")

print("\n⚠️  NOTE: Full weakness resolution requires running agent with candidates")
print("from weak positions. This would be tested by example_run.py per-position tests.")

print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)
