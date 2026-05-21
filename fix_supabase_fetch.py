from pathlib import Path
import re

project_root = Path.cwd()
webapp_path = project_root / 'webapp.py'

if not webapp_path.exists():
    raise FileNotFoundError('webapp.py not found in current directory')

text = webapp_path.read_text(encoding='utf-8')
backup_path = webapp_path.with_suffix('.py.bak_supabase')
backup_path.write_text(text, encoding='utf-8')

new_func = '''def get_supabase_image_url(image_path):
    """Convert FAISS/local image path to Supabase public URL.
    Expected bucket structure: fashion-images/<category>/<filename>
    Works even when FAISS stores old Windows absolute paths.
    """
    path_str = str(image_path).replace("\\", "/")

    if "/images/" in path_str:
        relative = path_str.split("/images/", 1)[1]
    elif "images/" in path_str:
        relative = path_str.split("images/", 1)[1]
    else:
        relative = Path(path_str).name

    relative = relative.lstrip("/")
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{relative}"
'''

pattern = r'def get_supabase_image_url\(image_path\):.*?\n(?=def |@app\.route|HTML_TEMPLATE\s*=|if __name__ == .__main__.:)'

if re.search(pattern, text, flags=re.S):
    updated = re.sub(pattern, new_func + '\n', text, flags=re.S)
else:
    marker = 'search_engine = None\n'
    if marker in text:
        updated = text.replace(marker, marker + '\n' + new_func + '\n')
    else:
        raise RuntimeError('Could not find existing get_supabase_image_url() or insertion marker in webapp.py')

if 'from pathlib import Path' not in updated:
    updated = 'from pathlib import Path\n' + updated

webapp_path.write_text(updated, encoding='utf-8')
print(f'Patched: {webapp_path}')
print(f'Backup created: {backup_path}')
print('New Supabase URL logic uses the path segment after images/.')
