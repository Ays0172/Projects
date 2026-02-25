"""Fix the broken init block in index.html"""
import re

filepath = r'c:\Users\AyushSood\Desktop\Portfolio01\Python_Portfolio\Projects\Employee Data Cleaner\Django\cleaner\templates\cleaner\index.html'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the broken init block
old_block = re.search(
    r'// ── Page-load init.*?\}\s*\)\s*\(\s*\)\s*;',
    content, re.DOTALL
)

if old_block:
    new_block = """// ── Page-load init: populate Step 3 dropdowns if data exists ────
    (function() {
      {%% if columns_json %%}
      var _cols = {{ columns_json|safe }};
      var _types = {{ dtypes_json|safe }};
      if (_cols && _cols.length > 0) {
        var h = '<div class="form-row">';
        for (var i = 0; i < _cols.length; i++) {
          var c = _cols[i];
          var t = (_types && _types[c]) || 'object';
          h += '<div style="flex:1 1 200px"><label title="Current: '+t+'">'+c+' <span style="font-size:0.65rem">('+t+')</span></label><select class="type-convert-select" data-col="'+c+'"><option value="none" selected>Skip (Keep '+t+')</option><option value="int">Integer</option><option value="float">Float</option><option value="str">String</option><option value="bool">Boolean</option><option value="datetime">Datetime</option></select></div>';
        }
        h += '</div>';
        document.getElementById("convert-options").innerHTML = h;
      }
      {%% endif %%}
    })();"""
    # The %% is for Python string formatting - we don't need it here since we're not using %
    # Actually let me just use a raw replacement
    print(f"Found block at positions {old_block.start()}-{old_block.end()}")
    print(f"Old block preview: {old_block.group()[:100]}...")
else:
    print("Block not found!")

# Let's just do a simple line-by-line fix instead
lines = content.split('\n')
new_lines = []
skip_until_end = False
found = False

for i, line in enumerate(lines):
    if '// ── Page-load init' in line:
        found = True
        skip_until_end = True
        # Write the new block
        new_lines.append('    // ── Page-load init: populate Step 3 dropdowns if data exists ────')
        new_lines.append('    (function() {')
        new_lines.append('      {% if columns_json %}')
        new_lines.append('      var _cols = {{ columns_json|safe }};')
        new_lines.append('      var _types = {{ dtypes_json|safe }};')
        new_lines.append('      if (_cols && _cols.length > 0) {')
        new_lines.append("        var h = '<div class=\"form-row\">';")
        new_lines.append('        for (var i = 0; i < _cols.length; i++) {')
        new_lines.append('          var c = _cols[i];')
        new_lines.append("          var t = (_types && _types[c]) || 'object';")
        new_lines.append("          h += '<div style=\"flex:1 1 200px\"><label title=\"Current: '+t+'\">'+c+' <span style=\"font-size:0.65rem\">('+t+')</span></label><select class=\"type-convert-select\" data-col=\"'+c+'\"><option value=\"none\" selected>Skip (Keep '+t+')</option><option value=\"int\">Integer</option><option value=\"float\">Float</option><option value=\"str\">String</option><option value=\"bool\">Boolean</option><option value=\"datetime\">Datetime</option></select></div>';")
        new_lines.append('        }')
        new_lines.append("        h += '</div>';")
        new_lines.append('        document.getElementById("convert-options").innerHTML = h;')
        new_lines.append('      }')
        new_lines.append('      {% endif %}')
        new_lines.append('    })();')
        continue
    
    if skip_until_end:
        # Skip lines until we find the end of the old block: }) ();
        stripped = line.strip()
        if stripped in ['}) ();', '})();', '}) ()', '})()']:
            skip_until_end = False
            continue
        else:
            continue
    
    new_lines.append(line)

if found:
    new_content = '\n'.join(new_lines)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("SUCCESS: Init block replaced!")
else:
    print("ERROR: Could not find init block!")
