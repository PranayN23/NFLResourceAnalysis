#!/usr/bin/env python3
from tools import evaluate_player

print("\n--- Testing Defensive Position Wrappers ---\n")

# Test LB player
lb_result = evaluate_player('Bobby Wagner')
print("LB Player (Bobby Wagner):")
print(f"  Position: {lb_result.get('position')}")
print(f"  Tier: {lb_result.get('tier')}")
print(f"  Grade: {lb_result.get('predicted_grade')}")
print(f"  Source: {lb_result.get('confidence', {}).get('source', 'N/A')}")

# Test CB player
cb_result = evaluate_player('Stephon Gilmore')
print("\nCB Player (Stephon Gilmore):")
print(f"  Position: {cb_result.get('position')}")
print(f"  Tier: {cb_result.get('tier')}")
print(f"  Grade: {cb_result.get('predicted_grade')}")
print(f"  Source: {cb_result.get('confidence', {}).get('source', 'N/A')}")

# Test S player
s_result = evaluate_player('Jamal Adams')
print("\nS Player (Jamal Adams):")
print(f"  Position: {s_result.get('position')}")
print(f"  Tier: {s_result.get('tier')}")
print(f"  Grade: {s_result.get('predicted_grade')}")
print(f"  Source: {s_result.get('confidence', {}).get('source', 'N/A')}")

print("\n✓ All defensive position wrappers successfully loaded and used!")
