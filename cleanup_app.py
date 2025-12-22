
import os

path = "tanml/ui/app.py"
with open(path, "r") as f:
    lines = f.readlines()

# Markers
start_marker = "# UI — Refit-only (20 models)"
end_marker = "def render_model_form"

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if start_marker in line:
        # Move up to the separator line
        if i > 0 and "==========" in lines[i-1]:
            start_idx = i - 1
        else:
            start_idx = i
        break

if start_idx != -1:
    for i in range(start_idx, len(lines)):
        if end_marker in lines[i]:
            end_idx = i
            break

if start_idx != -1 and end_idx != -1:
    print(f"Removing lines {start_idx} to {end_idx}")
    # Keep lines before start_idx and lines from end_idx onwards
    new_lines = lines[:start_idx] + lines[end_idx:]
    
    with open(path, "w") as f:
        f.writelines(new_lines)
    print("Cleanup successful.")
else:
    print("Markers not found!")
