import os
import uuid
import random
import logging
import gc
import shutil
import urllib.parse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_from_directory, session
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RENDER_DATA_DIR = Path("/opt/render/project/data")
DATA_DIR = RENDER_DATA_DIR if RENDER_DATA_DIR.exists() else BASE_DIR
MODEL_PATH = DATA_DIR / "models" / "clip_vit_b16_finetuned.pth"
INDEX_DIR = BASE_DIR / "search_index"
IMAGES_DIR = BASE_DIR / "images"


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "luxe-dev-secret-key")
CORS(app)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

UPLOAD_FOLDER  = os.path.join('static', 'uploads')
RESULTS_FOLDER = os.path.join('static', 'results')
os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Supabase
SUPABASE_URL    = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY    = os.environ.get("SUPABASE_KEY", "")
SUPABASE_BUCKET = "fashion-images"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Search engine (lazy-loaded)
search_engine = None

def get_supabase_image_url(image_path):
    path_str = str(image_path).replace("\\", "/")

    if "/images/" in path_str:
        relative = path_str.split("/images/", 1)[1]
    elif path_str.startswith("images/"):
        relative = path_str.split("images/", 1)[1]
    else:
        relative = path_str

    relative = relative.lstrip("/")
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{relative}"


def get_search_engine():
    global search_engine
    if search_engine is None:
        from visual_search import VisualSearchEngine
        search_engine = VisualSearchEngine(
            index_folder=str(INDEX_DIR),
            model_path=str(MODEL_PATH)
        )
    return search_engine

# Session helpers
def get_store(name, default):
    if name not in session:
        session[name] = default
    return session[name]

def add_history_entry(query_type, query_text, results_count, search_keyword):
    history = get_store('history_items', [])
    history.insert(0, {
        'id':             str(uuid.uuid4()),
        'query_type':     query_type,
        'query_text':     query_text,
        'results_count':  int(results_count),
        'search_keyword': search_keyword,
        'created_at':     datetime.now().strftime('%d %b %Y, %I:%M %p')
    })
    session['history_items'] = history[:50]
    session.modified = True

def get_supabase_image_url(image_path: str) -> str:
    images_root = str(IMAGES_DIR)
    try:
        relative = os.path.relpath(image_path, images_root).replace("\\", "/")
    except ValueError:
        relative = Path(image_path).name
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{relative}"

def extract_product_keywords(filename: str) -> str:
    base = filename.lower().replace('.jpg','').replace('.png','').replace('result_','')
    categories = {
        'shirt':   ['shirt','formal','casual'],
        'tshirt':  ['tshirt','t-shirt'],
        'jeans':   ['jeans','denim','pants'],
        'dress':   ['dress','frock','gown'],
        'kurta':   ['kurta','ethnic','traditional'],
        'saree':   ['saree','sari'],
        'jacket':  ['jacket','blazer','coat'],
        'shoes':   ['shoes','footwear','sneakers'],
        'jordan':  ['jordan'],
        'nike':    ['nike'],
        'adidas':  ['adidas'],
        'skirt':   ['skirt'],
        'trouser': ['trouser','pant'],
    }
    for category, keywords in categories.items():
        if any(kw in base for kw in keywords):
            return category
    return 'clothing'

def generate_ecommerce_links(search_keyword: str) -> dict:
    kw_enc    = urllib.parse.quote_plus(search_keyword)
    kw_myntra = urllib.parse.quote(search_keyword)
    return {
        "flipkart": f"https://www.flipkart.com/search?q={kw_enc}",
        "amazon":   f"https://www.amazon.in/s?k={kw_enc}",
        "myntra":   f"https://www.myntra.com/{kw_myntra}",
        "ajio":     f"https://www.ajio.com/search/?text={kw_enc}",
        "meesho":   f"https://www.meesho.com/search?q={kw_enc}",
        "nykaa":    f"https://www.nykaafashion.com/search/result/?q={kw_enc}",
    }

def render_page(active_page):
    history = get_store('history_items', [])
    saved   = get_store('saved_items',   [])
    return render_template_string(HTML_TEMPLATE, active_page=active_page, history=history, saved=saved)

@app.route('/')
def index():
    return render_page('home')

@app.route('/search-page')
def search_page():
    return render_page('search')

@app.route('/history-page')
def history_page():
    return render_page('history')

@app.route('/saved-page')
def saved_page():
    return render_page('saved')

@app.route('/search', methods=['POST'])
def search():
    try:
        engine     = get_search_engine()
        query_type = request.form.get('query_type', 'text')
        k          = int(request.form.get('k', 6))
        results    = []
        search_keyword = ''
        query_label    = ''

        if query_type == 'text':
            text_query = request.form.get('text_query', '').strip()
            if not text_query:
                return jsonify({"error": "Text query is required"}), 400
            search_keyword = text_query
            query_label    = text_query
            logger.info(f"Text search: {text_query}")
            results = engine.search_by_text(text_query, k)

        elif query_type == 'image':
            if 'image_query' not in request.files:
                return jsonify({"error": "Image file is required"}), 400
            file = request.files['image_query']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            query_label    = file.filename
            search_keyword = extract_product_keywords(file.filename)
            ext            = os.path.splitext(file.filename)[1]
            upload_name    = str(uuid.uuid4()) + ext
            file_path      = os.path.join(UPLOAD_FOLDER, upload_name)
            file.save(file_path)
            logger.info(f"Image search: {upload_name}")
            results = engine.search_by_image(file_path, k)

        formatted = []
        for i, result in enumerate(results):
            img_path     = result["image_path"]
            img_filename = Path(img_path).name
            result_filename = f"result_{i}_{img_filename}"
            result_path     = os.path.join(RESULTS_FOLDER, result_filename)
            try:
                shutil.copy2(img_path, result_path)
            except Exception as e:
                logger.error(f"Could not copy result {i+1}: {e}")

            image_url = get_supabase_image_url(img_path)

            print("DEBUG img_path:", img_path)
            print("DEBUG supabase_url:", image_url)

            base_price     = random.randint(3500, 8500)
            price_var      = int((1 - result['similarity']) * 1500)
            dummy_price    = base_price + price_var
            discount       = random.randint(5, 25)
            original_price = int(dummy_price / (1 - discount / 100))

            link_keyword = search_keyword
            if query_type == 'image' and search_keyword == 'clothing':
                link_keyword = extract_product_keywords(img_filename)

            formatted.append({
                "image_url":       image_url,
                "similarity":      f"{result['similarity']:.4f}",
                "filename":        img_filename,
                "price":           dummy_price,
                "original_price":  original_price,
                "discount":        discount,
                "category":        extract_product_keywords(img_filename),
                "search_keyword":  link_keyword,
                "ecommerce_links": generate_ecommerce_links(link_keyword),
            })

        add_history_entry(query_type, query_label, len(formatted), search_keyword)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"Returning {len(formatted)} results")
        return jsonify({"results": formatted})

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/saved', methods=['GET'])
def api_saved():
    return jsonify({'saved': get_store('saved_items', [])})

@app.route('/api/saved', methods=['POST'])
def api_saved_add():
    data      = request.get_json(force=True)
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({'error': 'image_url is required'}), 400
    saved = get_store('saved_items', [])
    if any(item['image_url'] == image_url for item in saved):
        return jsonify({'status': 'exists'})
    saved.insert(0, {
        'id':             str(uuid.uuid4()),
        'image_url':      image_url,
        'filename':       data.get('filename', 'fashion-item'),
        'price':          int(data.get('price', 0)),
        'original_price': int(data.get('original_price', 0)),
        'discount':       int(data.get('discount', 0)),
        'category':       data.get('category', 'clothing'),
        'search_keyword': data.get('search_keyword', 'fashion'),
        'created_at':     datetime.now().strftime('%d %b %Y, %I:%M %p'),
    })
    session['saved_items'] = saved[:100]
    session.modified = True
    return jsonify({'status': 'saved'})

@app.route('/api/saved/<item_id>', methods=['DELETE'])
def api_saved_delete(item_id):
    saved = get_store('saved_items', [])
    session['saved_items'] = [i for i in saved if i['id'] != item_id]
    session.modified = True
    return jsonify({'status': 'success'})

@app.route('/api/history', methods=['GET'])
def api_history():
    return jsonify({'history': get_store('history_items', [])})

@app.route('/api/history/clear', methods=['POST'])
def api_history_clear():
    session['history_items'] = []
    session.modified = True
    return jsonify({'status': 'success'})

@app.route('/api/history/<item_id>', methods=['DELETE'])
def api_history_delete(item_id):
    history = get_store('history_items', [])
    session['history_items'] = [h for h in history if h['id'] != item_id]
    session.modified = True
    return jsonify({'status': 'success'})

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LUXE &bull; Visual Search</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--primary:#1a1a1a;--secondary:#f8f8f8;--accent:#c9a96e;--text:#2c2c2c;--text-light:#6b7280;--border:#e5e5e5;--white:#ffffff;--shadow:0 8px 32px rgba(0,0,0,.08);--shadow-hover:0 16px 48px rgba(0,0,0,.12)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:var(--secondary);color:var(--text);line-height:1.6}
.container{max-width:1400px;margin:0 auto;padding:0 24px}
.header{background:var(--white);border-bottom:1px solid var(--border);padding:24px 0;margin-bottom:48px;position:sticky;top:0;z-index:100;box-shadow:var(--shadow)}
.header-inner{display:flex;align-items:center;justify-content:space-between}
.brand h1{font-family:'Playfair Display',serif;font-size:2rem;font-weight:600;color:var(--primary);letter-spacing:2px;cursor:pointer}
.brand .tagline{font-size:.75rem;color:var(--text-light);font-weight:300;letter-spacing:1px;text-transform:uppercase}
nav{display:flex;gap:4px}
.nav-link{padding:8px 18px;font-size:.85rem;font-weight:500;color:var(--text-light);text-decoration:none;border-radius:8px;letter-spacing:.3px;transition:all .2s ease;border:1px solid transparent}
.nav-link:hover{background:var(--secondary);color:var(--text)}
.nav-link.active{background:var(--primary);color:var(--white);border-color:var(--primary)}
.home-hero{text-align:center;padding:80px 24px 40px}
.home-hero h2{font-family:'Playfair Display',serif;font-size:clamp(2rem,5vw,3.5rem);font-weight:500;color:var(--primary);letter-spacing:1px;margin-bottom:16px}
.home-hero p{color:var(--text-light);font-size:1.05rem;font-weight:300;max-width:480px;margin:0 auto 32px}
.hero-cta{display:inline-block;padding:16px 40px;background:var(--primary);color:var(--white);text-decoration:none;border-radius:12px;font-size:.9rem;font-weight:500;letter-spacing:1px;text-transform:uppercase;transition:all .3s ease}
.hero-cta:hover{background:#333;transform:translateY(-2px);box-shadow:var(--shadow-hover)}
.search-section{background:var(--white);border-radius:16px;padding:40px;box-shadow:var(--shadow);margin-bottom:48px;max-width:600px;margin-left:auto;margin-right:auto}
.section-title{font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:500;color:var(--primary);margin-bottom:28px;letter-spacing:.5px}
.form-group{margin-bottom:24px}
.form-label{display:block;font-size:.8rem;font-weight:600;color:var(--primary);margin-bottom:8px;text-transform:uppercase;letter-spacing:.8px}
.form-control,.form-select{width:100%;padding:14px 18px;border:2px solid var(--border);border-radius:10px;font-size:.95rem;background:var(--white);transition:all .3s ease;font-family:inherit;color:var(--text)}
.form-control:focus,.form-select:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 4px rgba(201,169,110,.1)}
.search-btn{width:100%;padding:16px;background:var(--primary);color:var(--white);border:none;border-radius:10px;font-size:.9rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;cursor:pointer;transition:all .3s ease;font-family:inherit}
.search-btn:hover:not(:disabled){background:#333;transform:translateY(-1px);box-shadow:var(--shadow-hover)}
.search-btn:disabled{opacity:.6;cursor:not-allowed}
.loading{text-align:center;padding:80px 20px}
.loading-spinner{width:32px;height:32px;border:3px solid var(--border);border-top:3px solid var(--accent);border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 20px}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-text{color:var(--text-light);font-weight:300}
.results-section{margin-top:48px}
.results-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:32px;padding-bottom:16px;border-bottom:1px solid var(--border)}
.results-title{font-family:'Playfair Display',serif;font-size:1.8rem;font-weight:500;color:var(--primary);letter-spacing:1px}
.results-count{font-size:.85rem;color:var(--text-light)}
.results-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:28px}
.result-card{background:var(--white);border-radius:16px;overflow:hidden;box-shadow:var(--shadow);transition:all .4s ease;border:1px solid var(--border)}
.result-card:hover{transform:translateY(-8px);box-shadow:var(--shadow-hover)}
.result-image{width:100%;height:280px;object-fit:cover;background:var(--secondary)}
.card-content{padding:20px 24px 24px}
.card-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.similarity-badge{background:linear-gradient(135deg,var(--accent) 0%,#d4b377 100%);color:var(--white);padding:5px 12px;border-radius:20px;font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.save-btn{background:none;border:1.5px solid var(--border);width:36px;height:36px;border-radius:50%;font-size:1rem;cursor:pointer;display:flex;align-items:center;justify-content:center;color:var(--text-light);transition:all .3s ease}
.save-btn:hover{border-color:#dc2626;color:#dc2626;transform:scale(1.1)}
.save-btn.saved{background:#fef2f2;border-color:#dc2626;color:#dc2626}
.price-section{margin:16px 0;padding:16px 0;border-top:1px solid var(--border);border-bottom:1px solid var(--border)}
.current-price{font-size:1.5rem;font-weight:700;color:var(--primary)}
.original-price{text-decoration:line-through;color:var(--text-light);font-size:.95rem;margin-left:10px}
.discount-badge{background:#dc2626;color:var(--white);padding:2px 8px;border-radius:6px;font-size:.7rem;font-weight:600;margin-left:10px}
.shop-section{margin-top:16px}
.shop-title{font-size:.75rem;font-weight:600;color:var(--text-light);margin-bottom:10px;text-transform:uppercase;letter-spacing:.8px}
.shop-links{display:flex;flex-wrap:wrap;gap:6px}
.shop-link{padding:6px 14px;background:var(--secondary);color:var(--text);text-decoration:none;border-radius:8px;font-size:.78rem;font-weight:500;transition:all .25s ease;border:1px solid var(--border)}
.shop-link:hover{color:var(--white);transform:translateY(-1px)}
.shop-link[data-platform="flipkart"]:hover{background:#2874f0;border-color:#2874f0}
.shop-link[data-platform="amazon"]:hover{background:#ff9900;border-color:#ff9900;color:#111}
.shop-link[data-platform="myntra"]:hover{background:#ff3f6c;border-color:#ff3f6c}
.shop-link[data-platform="ajio"]:hover{background:#e8173d;border-color:#e8173d}
.shop-link[data-platform="meesho"]:hover{background:#9b2b97;border-color:#9b2b97}
.shop-link[data-platform="nykaa"]:hover{background:#fc2779;border-color:#fc2779}
.page-content{max-width:900px;margin:0 auto;padding:48px 24px 80px}
.page-title{font-family:'Playfair Display',serif;font-size:2rem;font-weight:500;color:var(--primary);letter-spacing:1px;margin-bottom:8px}
.page-subtitle{color:var(--text-light);font-size:.9rem;font-weight:300;margin-bottom:32px}
.page-actions{display:flex;justify-content:flex-end;margin-bottom:20px}
.btn-ghost{background:none;border:1px solid var(--border);padding:8px 18px;border-radius:8px;font-size:.8rem;font-weight:500;color:var(--text-light);cursor:pointer;transition:all .2s ease;font-family:inherit}
.btn-ghost:hover{border-color:#dc2626;color:#dc2626}
.history-list{display:flex;flex-direction:column;gap:10px}
.history-item{background:var(--white);border-radius:12px;padding:16px 20px;border:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;transition:box-shadow .2s ease}
.history-item:hover{box-shadow:var(--shadow)}
.history-left{flex:1}
.history-query{font-weight:500;font-size:.95rem;color:var(--text);margin-bottom:4px}
.history-meta{font-size:.78rem;color:var(--text-light)}
.history-right{display:flex;align-items:center;gap:12px;flex-shrink:0}
.history-count{font-size:.82rem;color:var(--accent);font-weight:600;white-space:nowrap}
.type-badge{display:inline-block;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.8px;padding:2px 8px;border-radius:4px;margin-right:8px}
.type-text{background:rgba(201,169,110,.15);color:var(--accent)}
.type-image{background:rgba(37,99,235,.1);color:#2563eb}
.delete-btn{background:none;border:none;font-size:1rem;cursor:pointer;color:var(--border);transition:color .2s ease;padding:2px 4px}
.delete-btn:hover{color:#dc2626}
.saved-card-meta{font-size:.75rem;color:var(--text-light);margin-top:8px;font-style:italic}
.alert{padding:18px 20px;border-radius:12px;margin:20px 0;font-weight:500;font-size:.9rem}
.alert-error{background:#fef2f2;color:#dc2626;border:1px solid #fecaca}
.alert-info{background:#f0f9ff;color:#0369a1;border:1px solid #bae6fd}
.empty-state{text-align:center;padding:80px 24px}
.empty-icon{font-size:2.5rem;margin-bottom:16px}
.empty-state p{color:var(--text-light);font-size:.95rem;font-weight:300;margin-bottom:24px}
@media(max-width:768px){.container{padding:0 16px}.header-inner{flex-direction:column;gap:16px}.brand h1{font-size:1.6rem}.search-section{padding:24px}.results-grid{grid-template-columns:1fr;gap:20px}.result-image{height:240px}nav{flex-wrap:wrap;justify-content:center}}
.fade-in{animation:fadeIn .5s ease-out forwards}
@keyframes fadeIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
</style>
</head>
<body>

<div class="header">
  <div class="container">
    <div class="header-inner">
      <div class="brand">
        <h1 onclick="window.location='/'">LUXE</h1>
        <p class="tagline">Premium Visual Search</p>
      </div>
      <nav>
        <a class="nav-link {{ 'active' if active_page=='home' else '' }}" href="/">Home</a>
        <a class="nav-link {{ 'active' if active_page=='search' else '' }}" href="/search-page">Search</a>
        <a class="nav-link {{ 'active' if active_page=='history' else '' }}" href="/history-page">History</a>
        <a class="nav-link {{ 'active' if active_page=='saved' else '' }}" href="/saved-page">Saved</a>
      </nav>
    </div>
  </div>
</div>

{% if active_page == 'home' %}
<div class="container">
  <div class="home-hero">
    <h2>Discover Your Style</h2>
    <p>AI-powered visual search — find exactly what you're looking for across India's top fashion platforms.</p>
    <a href="/search-page" class="hero-cta">Start Searching</a>
  </div>
</div>
{% endif %}

{% if active_page == 'search' %}
<div class="container">
  <div class="search-section">
    <h2 class="section-title">Find Your Look</h2>
    <div class="form-group">
      <label class="form-label">Search Method</label>
      <select id="queryType" class="form-select">
        <option value="text">Text Search</option>
        <option value="image">Image Search</option>
      </select>
    </div>
    <div class="form-group" id="textQueryGroup">
      <label class="form-label">Describe Your Style</label>
      <input type="text" id="textQuery" class="form-control" placeholder="e.g., elegant black dress, white sneakers...">
    </div>
    <div class="form-group" id="imageQueryGroup" style="display:none">
      <label class="form-label">Upload Image</label>
      <input type="file" id="imageQuery" class="form-control" accept="image/*">
    </div>
    <div class="form-group">
      <label class="form-label">Results Count</label>
      <input type="number" id="resultsK" value="6" min="1" max="12" class="form-control">
    </div>
    <button type="button" class="search-btn" id="searchBtn" onclick="doSearch()">Discover</button>
  </div>

  <div id="loading" class="loading" style="display:none">
    <div class="loading-spinner"></div>
    <p class="loading-text">Curating items for you...</p>
  </div>
  <div id="results"></div>
</div>

<script>
// toggle input fields
document.getElementById('queryType').addEventListener('change', function() {
  var isText = this.value === 'text';
  document.getElementById('textQueryGroup').style.display  = isText ? 'block' : 'none';
  document.getElementById('imageQueryGroup').style.display = isText ? 'none'  : 'block';
});

function doSearch() {
  var queryType = document.getElementById('queryType').value;
  var k         = document.getElementById('resultsK').value || '6';
  var btn       = document.getElementById('searchBtn');

  // build FormData manually (no <form> tag needed)
  var fd = new FormData();
  fd.append('query_type', queryType);
  fd.append('k', k);

  if (queryType === 'text') {
    var q = document.getElementById('textQuery').value.trim();
    if (!q) { alert('Please enter a search query.'); return; }
    fd.append('text_query', q);
  } else {
    var fileInput = document.getElementById('imageQuery');
    if (!fileInput.files || fileInput.files.length === 0) {
      alert('Please select an image file.');
      return;
    }
    fd.append('image_query', fileInput.files[0]);
  }

  btn.disabled    = true;
  btn.textContent = 'Searching...';
  document.getElementById('loading').style.display = 'block';
  document.getElementById('results').innerHTML     = '';

  fetch('/search', { method: 'POST', body: fd })
    .then(function(r) {
      if (!r.ok) throw new Error('Server error ' + r.status);
      return r.json();
    })
    .then(function(data) {
      document.getElementById('loading').style.display = 'none';
      btn.disabled    = false;
      btn.textContent = 'Discover';
      if (data.error) {
        document.getElementById('results').innerHTML =
          '<div class="alert alert-error">' + data.error + '</div>';
        return;
      }
      displayResults(data.results);
    })
    .catch(function(err) {
      document.getElementById('loading').style.display = 'none';
      btn.disabled    = false;
      btn.textContent = 'Discover';
      document.getElementById('results').innerHTML =
        '<div class="alert alert-error">Request failed: ' + err.message + '</div>';
    });
}

var PLATFORM_LABELS = {
  flipkart:'Flipkart', amazon:'Amazon', myntra:'Myntra',
  ajio:'Ajio', meesho:'Meesho', nykaa:'Nykaa Fashion'
};

function displayResults(items) {
  var container = document.getElementById('results');
  if (!items || items.length === 0) {
    container.innerHTML = '<div class="alert alert-info">No items found.</div>';
    return;
  }
  var cards = items.map(function(r, i) {
    var links = '';
    Object.keys(r.ecommerce_links).forEach(function(p) {
      links += '<a href="' + r.ecommerce_links[p] + '" target="_blank" rel="noopener" ' +
               'class="shop-link" data-platform="' + p + '">' +
               (PLATFORM_LABELS[p] || p) + '</a>';
    });
    var enc = encodeURIComponent(JSON.stringify(r));
    return '<div class="result-card fade-in" style="animation-delay:' + (i * 0.08) + 's">' +
      '<img src="' + r.image_url + '" class="result-image" ' +
      'onerror="this.src=\'https://placehold.co/300x280/f8f8f8/aaa?text=Image\'">' +
      '<div class="card-content">' +
        '<div class="card-top">' +
          '<span class="similarity-badge">' + (parseFloat(r.similarity)*100).toFixed(1) + '% Match</span>' +
          '<button type="button" class="save-btn" data-item="' + enc + '" onclick="saveItem(this)">&#9825;</button>' +
        '</div>' +
        '<div class="price-section">' +
          '<span class="current-price">&#8377;' + r.price.toLocaleString('en-IN') + '</span>' +
          '<span class="original-price">&#8377;' + r.original_price.toLocaleString('en-IN') + '</span>' +
          '<span class="discount-badge">' + r.discount + '% OFF</span>' +
        '</div>' +
        '<div class="shop-section"><div class="shop-title">Available At</div>' +
          '<div class="shop-links">' + links + '</div>' +
        '</div>' +
      '</div></div>';
  }).join('');

  container.innerHTML =
    '<div class="results-section">' +
      '<div class="results-header">' +
        '<h2 class="results-title">Curated Collection</h2>' +
        '<span class="results-count">' + items.length + ' items found</span>' +
      '</div>' +
      '<div class="results-grid">' + cards + '</div>' +
    '</div>';
}

function saveItem(btn) {
  var item = JSON.parse(decodeURIComponent(btn.dataset.item));
  fetch('/api/saved', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(item)
  })
  .then(function(r){ return r.json(); })
  .then(function(d){
    if (d.status === 'saved' || d.status === 'exists') {
      btn.classList.add('saved');
      btn.innerHTML = '&#9829;';
      btn.title = 'Saved!';
    }
  }).catch(console.error);
}
</script>
{% endif %}

{% if active_page == 'history' %}
<div class="page-content">
  <h2 class="page-title">Search History</h2>
  <p class="page-subtitle">Your recent searches, auto-saved this session.</p>
  {% if history %}
  <div class="page-actions">
    <button class="btn-ghost" onclick="clearHistory()">Clear All</button>
  </div>
  <div class="history-list" id="historyList">
    {% for h in history %}
    <div class="history-item" id="hist-{{ h.id }}">
      <div class="history-left">
        <div class="history-query">
          <span class="type-badge {{ 'type-text' if h.query_type=='text' else 'type-image' }}">{{ h.query_type }}</span>
          {{ h.query_text }}
        </div>
        <div class="history-meta">{{ h.created_at }} &nbsp;&middot;&nbsp; keyword: {{ h.search_keyword }}</div>
      </div>
      <div class="history-right">
        <span class="history-count">{{ h.results_count }} results</span>
        <button class="delete-btn" onclick="deleteHistory('{{ h.id }}')" title="Remove">&#10005;</button>
      </div>
    </div>
    {% endfor %}
  </div>
  <script>
  function clearHistory() {
    if (!confirm('Clear all search history?')) return;
    fetch('/api/history/clear', {method:'POST'})
      .then(function(){ window.location.reload(); }).catch(console.error);
  }
  function deleteHistory(id) {
    fetch('/api/history/' + id, {method:'DELETE'})
      .then(function(){
        var r = document.getElementById('hist-' + id);
        if (r) { r.style.opacity='0'; r.style.transition='opacity .3s'; setTimeout(function(){ r.remove(); }, 300); }
      }).catch(console.error);
  }
  </script>
  {% else %}
  <div class="empty-state">
    <div class="empty-icon">&#128269;</div>
    <p>No searches yet. Head to Search and discover something.</p>
    <a href="/search-page" class="hero-cta">Go to Search</a>
  </div>
  {% endif %}
</div>
{% endif %}

{% if active_page == 'saved' %}
<div class="page-content">
  <h2 class="page-title">Saved Items</h2>
  <p class="page-subtitle">Items you hearted this session.</p>
  {% if saved %}
  <div class="results-grid" id="savedGrid">
    {% for item in saved %}
    <div class="result-card fade-in" id="saved-{{ item.id }}">
      <img src="{{ item.image_url }}" alt="{{ item.filename }}" class="result-image"
           onerror="this.src='https://placehold.co/300x280/f8f8f8/aaa?text=Image'">
      <div class="card-content">
        <div class="card-top">
          <span class="similarity-badge">Saved</span>
          <button type="button" class="save-btn saved" onclick="removeSaved('{{ item.id }}')">&#9829;</button>
        </div>
        <div class="price-section">
          <span class="current-price">&#8377;{{ item.price|int }}</span>
          <span class="original-price">&#8377;{{ item.original_price|int }}</span>
          <span class="discount-badge">{{ item.discount }}% OFF</span>
        </div>
        <div class="shop-section">
          <div class="shop-title">Shop Now</div>
          <div class="shop-links">
            {% set kw = item.search_keyword %}
            <a href="https://www.flipkart.com/search?q={{ kw|urlencode }}" target="_blank" rel="noopener" class="shop-link" data-platform="flipkart">Flipkart</a>
            <a href="https://www.amazon.in/s?k={{ kw|urlencode }}" target="_blank" rel="noopener" class="shop-link" data-platform="amazon">Amazon</a>
            <a href="https://www.myntra.com/{{ kw }}" target="_blank" rel="noopener" class="shop-link" data-platform="myntra">Myntra</a>
            <a href="https://www.ajio.com/search/?text={{ kw|urlencode }}" target="_blank" rel="noopener" class="shop-link" data-platform="ajio">Ajio</a>
          </div>
        </div>
        <div class="saved-card-meta">{{ item.created_at }}</div>
      </div>
    </div>
    {% endfor %}
  </div>
  <script>
  function removeSaved(id) {
    fetch('/api/saved/' + id, {method:'DELETE'})
      .then(function(){
        var c = document.getElementById('saved-' + id);
        if (c) { c.style.opacity='0'; c.style.transform='scale(.95)'; c.style.transition='all .3s'; setTimeout(function(){ c.remove(); }, 300); }
      }).catch(console.error);
  }
  </script>
  {% else %}
  <div class="empty-state">
    <div class="empty-icon">&#9825;</div>
    <p>Nothing saved yet. Heart an item from your search results.</p>
    <a href="/search-page" class="hero-cta">Go to Search</a>
  </div>
  {% endif %}
</div>
{% endif %}

</body>
</html>"""

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8005)