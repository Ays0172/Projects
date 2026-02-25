"""
Fixes ALL broken Django template tags in index.html.
Run once, then restart the Django dev server.
"""
import re

filepath = r'c:\Users\AyushSood\Desktop\Portfolio01\Python_Portfolio\Projects\Employee Data Cleaner\Django\cleaner\templates\cleaner\index.html'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

original = content

# ── Fix the reset button specifically ──────────────────────────────────────────
content = re.sub(
    r'<button class="btn btn-danger btn-sm" onclick="resetPipeline\(\)" id="btn-reset"[^>]*>↺ Reset Pipeline</button>',
    '<button class="btn btn-danger btn-sm" onclick="resetPipeline()" id="btn-reset" {% if not summary %}disabled{% endif %}>↺ Reset Pipeline</button>',
    content
)

# ── Fix any remaining {%word (missing space after {%) ──────────────────────────
content = re.sub(r'\{%([a-z])', r'{% \1', content)

# ── Fix any remaining word%} (missing space before %}) ─────────────────────────
content = re.sub(r'([a-z])%\}', r'\1 %}', content)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

changed = content != original
print("FIXED" if changed else "NO CHANGE NEEDED")

for i, line in enumerate(content.split('\n'), 1):
    if 'btn-reset' in line:
        print(f"Reset button - Line {i}: {line.strip()}")
