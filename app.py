from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import math
import hashlib
import psycopg2
import psycopg2.extras
from datetime import datetime
from classifier import ExpenseClassifier
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-me-in-production')

# Cookie settings — must be Secure + SameSite=None for Railway HTTPS
# (SameSite=Lax is fine when frontend and backend share the same domain, which they do on Railway)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('RAILWAY_ENVIRONMENT') is not None
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24 * 30  # 30 days

CORS(app, supports_credentials=True)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
DATA_FOLDER   = 'data'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, DATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

classifier = ExpenseClassifier()

# ─────────────────────────────────────────────
#  Database helpers
# ─────────────────────────────────────────────

def get_db():
    url = os.environ.get('DATABASE_URL')
    if not url:
        raise RuntimeError('DATABASE_URL environment variable is not set')
    conn = psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        SERIAL PRIMARY KEY,
            email     TEXT UNIQUE NOT NULL,
            password  TEXT NOT NULL,
            name      TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS file_cache (
            file_hash   TEXT PRIMARY KEY,
            filename    TEXT NOT NULL,
            user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
            results_json TEXT NOT NULL,
            stats_json  TEXT NOT NULL,
            classified_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS classification_history (
            id          SERIAL PRIMARY KEY,
            user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
            filename    TEXT NOT NULL,
            file_hash   TEXT NOT NULL,
            total_rows  INTEGER,
            stats_json  TEXT,
            output_csv  TEXT,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database tables ready")


try:
    init_db()
except Exception as e:
    print(f"⚠️  DB init skipped (will retry on first request): {e}")


# ─────────────────────────────────────────────
#  Auth decorator
# ─────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required', 'auth_required': True}), 401
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def sanitize_value(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if isinstance(v, pd.Timestamp):
        return str(v)
    return v


def sanitize_records(records):
    return [{k: sanitize_value(v) for k, v in row.items()} for row in records]


def is_category_master_file(df):
    cols_lower = [str(c).lower().strip() for c in df.columns]
    col_str = ' '.join(cols_lower)
    master_signals = [
        'account name', 'account_name', 'accountname',
        'sub-group', 'subgroup', 'sub_group',
        'highlevel', 'high level', 'high_level',
        'sch iii', 'schedule', 'classification',
        'llm keyword', 'llmkeyword', 'llm keywords',
        'isactive', 'is_active', 'is active',
        'gst_flag', 'gstflag', 'tds_section',
        'boe_flag', 'repeating',
    ]
    hits = sum(1 for sig in master_signals if sig in col_str)
    return hits >= 2


def load_df(filepath, filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath, dtype=str)
    else:
        df = pd.read_excel(filepath, dtype=str, header=0)
    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
    df = df.dropna(how='all').fillna('')
    return df


def extract_categories_from_master(df):
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    priority = [
        'classification', 'account name', 'accountname', 'account_name',
        'highlevel classification', 'high level classification',
        'sub-group', 'subgroup', 'category', 'name'
    ]
    target_col = None
    for p in priority:
        if p in cols_lower:
            target_col = cols_lower[p]
            break
    if not target_col:
        for col in df.columns:
            if df[col].dtype == object:
                target_col = col
                break
    if not target_col:
        return []
    raw = df[target_col].astype(str).str.strip()
    categories = [
        v for v in raw.unique()
        if v and v.lower() not in ('nan', '', 'none')
        and not v.replace('.', '').replace('-', '').isdigit()
    ]
    return sorted(set(categories))


def build_stats(results, confidence_threshold):
    results['Confidence'] = pd.to_numeric(results['Confidence'], errors='coerce').fillna(0.0)
    category_totals, category_counts = {}, {}
    amount_col = 'Amount' if 'Amount' in results.columns else None
    for _, row in results.iterrows():
        cat = str(row.get('Category', 'Other') or 'Other')
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if amount_col:
            try:
                raw_amt = str(row[amount_col]).replace(',', '').replace('₹', '').replace('$', '').strip()
                amt = float(raw_amt)
                if not math.isnan(amt):
                    category_totals[cat] = category_totals.get(cat, 0) + abs(amt)
            except Exception:
                pass
    base = category_totals if category_totals else category_counts
    sorted_cats = dict(sorted(base.items(), key=lambda x: x[1], reverse=True))
    return {
        'total': int(len(results)),
        'high_confidence': int((results['Confidence'] >= confidence_threshold).sum()),
        'needs_review': int((results['Confidence'] < confidence_threshold).sum()),
        'ai_classified': int((results['Method'] == 'ai').sum()),
        'keyword_classified': int((results['Method'] == 'keyword').sum()),
        'category_totals': {str(k): float(v) for k, v in sorted_cats.items()},
        'category_counts': {str(k): int(v) for k, v in category_counts.items()},
    }


# ─────────────────────────────────────────────
#  Auth routes
# ─────────────────────────────────────────────

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json() or {}
    name     = (data.get('name') or '').strip()
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not name or not email or not password:
        return jsonify({'error': 'Name, email and password are required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute('SELECT id FROM users WHERE email=%s', (email,))
        if cur.fetchone():
            cur.close(); conn.close()
            return jsonify({'error': 'Email already registered'}), 409
        hashed = generate_password_hash(password)
        cur.execute(
            'INSERT INTO users (email, password, name) VALUES (%s,%s,%s) RETURNING id, name, email',
            (email, hashed, name)
        )
        user = dict(cur.fetchone())
        conn.commit(); cur.close(); conn.close()
        session.permanent = True
        session['user_id']    = user['id']
        session['user_name']  = user['name']
        session['user_email'] = user['email']
        return jsonify({'success': True, 'user': {'id': user['id'], 'name': user['name'], 'email': user['email']}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    data     = request.get_json() or {}
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute('SELECT id, name, email, password FROM users WHERE email=%s', (email,))
        user = cur.fetchone()
        cur.close(); conn.close()
        if not user or not check_password_hash(user['password'], password):
            return jsonify({'error': 'Invalid email or password'}), 401
        session.permanent = True
        session['user_id']    = user['id']
        session['user_name']  = user['name']
        session['user_email'] = user['email']
        return jsonify({'success': True, 'user': {'id': user['id'], 'name': user['name'], 'email': user['email']}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/api/auth/me', methods=['GET'])
def me():
    if 'user_id' not in session:
        return jsonify({'authenticated': False})
    return jsonify({
        'authenticated': True,
        'user': {'id': session['user_id'], 'name': session['user_name'], 'email': session['user_email']}
    })


# ─────────────────────────────────────────────
#  History routes
# ─────────────────────────────────────────────

@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            SELECT id, filename, file_hash, total_rows, stats_json, output_csv, created_at
            FROM classification_history
            WHERE user_id = %s
            ORDER BY created_at DESC LIMIT 50
        """, (session['user_id'],))
        rows = cur.fetchall()
        cur.close(); conn.close()
        history = []
        for r in rows:
            entry = dict(r)
            entry['stats']      = json.loads(entry.pop('stats_json') or '{}')
            entry['created_at'] = str(entry['created_at'])
            history.append(entry)
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/<int:history_id>/results', methods=['GET'])
@login_required
def get_history_results(history_id):
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            SELECT h.file_hash, h.filename, fc.results_json, fc.stats_json
            FROM classification_history h
            JOIN file_cache fc ON fc.file_hash = h.file_hash
            WHERE h.id = %s AND h.user_id = %s
        """, (history_id, session['user_id']))
        row = cur.fetchone()
        cur.close(); conn.close()
        if not row:
            return jsonify({'error': 'History entry not found'}), 404
        return jsonify({
            'success': True,
            'results': json.loads(row['results_json']),
            'stats':   json.loads(row['stats_json']),
            'filename': row['filename'],
            'from_cache': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/<int:history_id>/download', methods=['GET'])
@login_required
def download_history_file(history_id):
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            SELECT output_csv, filename FROM classification_history
            WHERE id = %s AND user_id = %s
        """, (history_id, session['user_id']))
        row = cur.fetchone()
        cur.close(); conn.close()
        if not row or not row['output_csv'] or not os.path.exists(row['output_csv']):
            return jsonify({'error': 'File not found'}), 404
        return send_file(row['output_csv'], mimetype='text/csv', as_attachment=True,
                         download_name=f"classified_{row['filename']}")
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
#  Core app routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload CSV or Excel'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        fhash = compute_file_hash(filepath)

        # Check cache for this user+file
        already_classified = False
        try:
            conn = get_db()
            cur  = conn.cursor()
            cur.execute('SELECT 1 FROM file_cache WHERE file_hash=%s AND user_id=%s',
                        (fhash, session['user_id']))
            already_classified = cur.fetchone() is not None
            cur.close(); conn.close()
        except Exception:
            pass

        df = load_df(filepath, filename)

        if is_category_master_file(df):
            df.to_csv(os.path.join(DATA_FOLDER, 'category_master.csv'), index=False)
            categories = extract_categories_from_master(df)
            classifier.load_categories_from_master(df, categories)
            preview = sanitize_records(df.head(8).to_dict('records'))
            return jsonify({
                'success': True,
                'file_mode': 'category_master',
                'filename': filename,
                'total_rows': len(df),
                'columns': list(df.columns),
                'preview': preview,
                'categories_loaded': len(categories),
                'categories_sample': categories[:20],
                'message': f'Category master loaded — {len(categories)} categories found.'
            })

        # Transactions
        df.to_csv(os.path.join(DATA_FOLDER, 'current_transactions.csv'), index=False)
        with open(os.path.join(DATA_FOLDER, 'current_hash.txt'), 'w') as f:
            f.write(f'{fhash}|||{filename}')

        file_type = classifier.detect_file_type(df)
        suggested_categories = classifier.get_auto_categories(file_type)
        preview = sanitize_records(df.head(5).to_dict('records'))

        response = {
            'success': True,
            'file_mode': 'transactions',
            'filename': filename,
            'file_hash': fhash,
            'total_rows': len(df),
            'columns': list(df.columns),
            'preview': preview,
            'file_type': file_type,
            'suggested_categories': suggested_categories,
            'has_category_master': classifier.categories_loaded,
            'already_classified': already_classified,
        }
        if already_classified:
            response['message'] = 'This file was already classified. Load previous results or re-classify.'
        return jsonify(response)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500


@app.route('/api/load-cached', methods=['POST'])
@login_required
def load_cached():
    try:
        hash_file = os.path.join(DATA_FOLDER, 'current_hash.txt')
        if not os.path.exists(hash_file):
            return jsonify({'error': 'No file uploaded yet'}), 400
        with open(hash_file) as f:
            fhash, filename = f.read().split('|||', 1)
        conn = get_db()
        cur  = conn.cursor()
        cur.execute('SELECT results_json, stats_json FROM file_cache WHERE file_hash=%s AND user_id=%s',
                    (fhash, session['user_id']))
        cached = cur.fetchone()
        cur.close(); conn.close()
        if not cached:
            return jsonify({'error': 'No cached results found'}), 404
        return jsonify({'success': True, 'results': json.loads(cached['results_json']),
                        'stats': json.loads(cached['stats_json']), 'from_cache': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/force-transactions', methods=['POST'])
@login_required
def force_transactions():
    try:
        master_file = os.path.join(DATA_FOLDER, 'category_master.csv')
        if not os.path.exists(master_file):
            return jsonify({'error': 'No master file found to convert'}), 400
        df = pd.read_csv(master_file, dtype=str).fillna('')
        df.to_csv(os.path.join(DATA_FOLDER, 'current_transactions.csv'), index=False)
        file_type = classifier.detect_file_type(df)
        return jsonify({'success': True, 'total_rows': len(df), 'file_type': file_type,
                        'message': f'File re-interpreted as transactions ({len(df)} rows)'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _get_current_hash():
    hash_file = os.path.join(DATA_FOLDER, 'current_hash.txt')
    if os.path.exists(hash_file):
        with open(hash_file) as f:
            parts = f.read().split('|||', 1)
            return parts[0], (parts[1] if len(parts) > 1 else 'transactions.csv')
    return None, 'transactions.csv'


def _persist_results(user_id, fhash, filename, results_safe, stats, output_file):
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO file_cache (file_hash, filename, user_id, results_json, stats_json)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (file_hash) DO UPDATE
            SET results_json=EXCLUDED.results_json,
                stats_json=EXCLUDED.stats_json,
                classified_at=NOW()
        """, (fhash, filename, user_id, json.dumps(results_safe), json.dumps(stats)))
        cur.execute("""
            INSERT INTO classification_history
                (user_id, filename, file_hash, total_rows, stats_json, output_csv)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (user_id, filename, fhash, stats['total'], json.dumps(stats), output_file))
        conn.commit(); cur.close(); conn.close()
    except Exception as db_err:
        print(f"⚠️  DB save failed: {db_err}")


@app.route('/api/classify-stream', methods=['GET'])
@login_required
def classify_stream():
    import queue, threading

    use_ai               = request.args.get('use_ai', 'true').lower() == 'true'
    confidence_threshold = float(request.args.get('confidence_threshold', 0.6))
    raw_cats             = request.args.get('categories', None)
    custom_categories    = json.loads(raw_cats) if raw_cats else None
    user_id              = session['user_id']
    fhash, fname         = _get_current_hash()

    trans_file = os.path.join(DATA_FOLDER, 'current_transactions.csv')
    if not os.path.exists(trans_file):
        def err_gen():
            yield 'data: ' + json.dumps({'error': 'No transactions file found'}) + '\n\n'
        return app.response_class(err_gen(), mimetype='text/event-stream')

    df = pd.read_csv(trans_file, dtype=str).fillna('')
    q  = queue.Queue()

    def worker():
        try:
            results, file_type, categories_used = classifier.classify_transactions(
                df, categories=custom_categories, use_ai=use_ai,
                confidence_threshold=confidence_threshold, progress_queue=q
            )
            timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(OUTPUT_FOLDER, f'classified_{timestamp}.csv')
            results.to_csv(output_file, index=False)
            stats = build_stats(results, confidence_threshold)
            stats['file_type'] = str(file_type)
            stats['categories_used'] = [str(c) for c in categories_used]
            results_safe = sanitize_records(results.where(pd.notnull(results), None).to_dict('records'))
            if fhash:
                _persist_results(user_id, fhash, fname, results_safe, stats, output_file)
            q.put({'type': 'done', 'results': results_safe, 'stats': stats,
                   'output_file': str(output_file)})
        except Exception as e:
            import traceback; traceback.print_exc()
            q.put({'type': 'error', 'error': str(e)})

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        yield 'data: ' + json.dumps({'type': 'start', 'total': len(df)}) + '\n\n'
        while True:
            try:
                msg = q.get(timeout=120)
            except Exception:
                yield 'data: ' + json.dumps({'type': 'error', 'error': 'Timeout'}) + '\n\n'
                return
            yield 'data: ' + json.dumps(msg) + '\n\n'
            if msg.get('type') in ('done', 'error'):
                return

    return app.response_class(generate(), mimetype='text/event-stream',
                               headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/classify', methods=['POST'])
@login_required
def classify_transactions_route():
    try:
        trans_file = os.path.join(DATA_FOLDER, 'current_transactions.csv')
        if not os.path.exists(trans_file):
            return jsonify({'error': 'Please upload a transactions file first'}), 400
        df       = pd.read_csv(trans_file, dtype=str).fillna('')
        req_data = request.get_json() or {}
        confidence_threshold = float(req_data.get('confidence_threshold', 0.6))
        use_ai           = req_data.get('use_ai', True)
        custom_categories = req_data.get('categories', None)
        user_id          = session['user_id']
        fhash, fname     = _get_current_hash()

        results, file_type, categories_used = classifier.classify_transactions(
            df, categories=custom_categories, use_ai=use_ai,
            confidence_threshold=confidence_threshold
        )
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_FOLDER, f'classified_{timestamp}.csv')
        results.to_csv(output_file, index=False)
        stats = build_stats(results, confidence_threshold)
        stats['file_type'] = str(file_type)
        stats['categories_used'] = [str(c) for c in categories_used]
        results_safe = sanitize_records(results.where(pd.notnull(results), None).to_dict('records'))
        if fhash:
            _persist_results(user_id, fhash, fname, results_safe, stats, output_file)
        return jsonify({'success': True, 'results': results_safe, 'stats': stats,
                        'output_file': str(output_file)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Classification error: {str(e)}'}), 500


@app.route('/api/update-classification', methods=['POST'])
@login_required
def update_classification():
    try:
        data = request.get_json()
        desc = data.get('description')
        cat  = data.get('category')
        if not desc or not cat:
            return jsonify({'error': 'Missing description or category'}), 400
        classifier.learn_mapping(desc, cat)
        return jsonify({'success': True, 'message': f'Learned: "{desc}" → {cat}'})
    except Exception as e:
        return jsonify({'error': f'Update error: {str(e)}'}), 500


@app.route('/api/download-results', methods=['GET'])
@login_required
def download_results():
    try:
        output_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith('classified_')]
        if not output_files:
            return jsonify({'error': 'No results available'}), 404
        filepath = os.path.join(OUTPUT_FOLDER, max(output_files))
        return send_file(filepath, mimetype='text/csv', as_attachment=True,
                         download_name='classified_transactions.csv')
    except Exception as e:
        return jsonify({'error': f'Download error: {str(e)}'}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        user_info = {'authenticated': 'user_id' in session}
        if 'user_id' in session:
            user_info.update({'user_name': session.get('user_name'), 'user_email': session.get('user_email')})
        return jsonify({
            'learned_mappings': int(len(classifier.vendor_memory)),
            'ai_provider':      str(classifier.ai_provider or 'none'),
            'ai_model':         str(getattr(classifier, 'openrouter_model', 'n/a')),
            'categories_loaded': bool(classifier.categories_loaded),
            'total_categories':  int(len(classifier.categories)) if classifier.categories_loaded else 0,
            **user_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
#  Timing endpoints
# ─────────────────────────────────────────────

_timing_store_file = os.path.join(DATA_FOLDER, 'timing_store.json')

def _load_timing():
    try:
        if os.path.exists(_timing_store_file):
            with open(_timing_store_file) as f:
                return json.load(f)
    except Exception:
        pass
    return {'ai': [], 'keyword': []}

def _save_timing(data):
    with open(_timing_store_file, 'w') as f:
        json.dump(data, f)

@app.route('/api/timing', methods=['GET'])
def get_timing():
    store  = _load_timing()
    result = {}
    for mode in ('ai', 'keyword'):
        samples = store.get(mode, [])
        if samples:
            recent = samples[-5:]
            result[mode] = round(sorted(recent)[len(recent) // 2], 4)
        else:
            result[mode] = 0.9 if mode == 'ai' else 0.08
    return jsonify(result)

@app.route('/api/timing', methods=['POST'])
def post_timing():
    try:
        data        = request.get_json() or {}
        rows        = int(data.get('rows', 0))
        elapsed_sec = float(data.get('elapsed_sec', 0))
        mode        = data.get('mode', 'ai')
        if rows <= 0 or elapsed_sec <= 0:
            return jsonify({'ok': False, 'reason': 'invalid data'}), 400
        spr = elapsed_sec / rows
        store = _load_timing()
        store.setdefault(mode, []).append(round(spr, 4))
        store[mode] = store[mode][-20:]
        _save_timing(store)
        return jsonify({'ok': True, 'sec_per_row': spr})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Smart Expense Classifier Starting on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port)