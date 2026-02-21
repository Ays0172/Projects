"""Quick script to fix the template syntax error on the reset button line."""
import re

filepath = r'c:\Users\AyushSood\Desktop\Portfolio01\Python_Portfolio\Projects\Employee Data Cleaner\Django\cleaner\templates\cleaner\index.html'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the reset button line - the {% if not summary %}disabled{% endif %} tag
# is split across lines. Use regex to find any variation and replace with single-line version.
pattern = r'id="btn-reset"\s*\{%\s*if\s+not\s+summary\s*%\}\s*disabled\s*\{%\s*endif\s*%\}'
replacement = 'id="btn-reset" {% if not summary %}disabled{% endif %}'

new_content = re.sub(pattern, replacement, content)

if new_content != content:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("FIXED: Reset button template tag corrected.")
else:
    print("No change needed or pattern not found.")
    # Let's search for the problematic area
    for i, line in enumerate(content.split('\n'), 1):
        if 'btn-reset' in line:
            print(f"  Line {i}: {repr(line)}")
