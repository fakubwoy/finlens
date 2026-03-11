from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import pandas as pd
import numpy as np
import json, os, math, hashlib
import psycopg2, psycopg2.extras
from datetime import datetime
from classifier import ExpenseClassifier
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-me-in-production')
app.config['SESSION_COOKIE_HTTPONLY']   = True
app.config['SESSION_COOKIE_SAMESITE']  = 'Lax'
app.config['SESSION_COOKIE_SECURE']    = os.environ.get('RAILWAY_ENVIRONMENT') is not None
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24 * 30
CORS(app, supports_credentials=True)

UPLOAD_FOLDER      = 'uploads'
OUTPUT_FOLDER      = 'outputs'
DATA_FOLDER        = 'data'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, DATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER']        = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']   = 16 * 1024 * 1024

classifier = ExpenseClassifier()

# ─────────────────────────────────────────────────────────────
#  DB helpers
# ─────────────────────────────────────────────────────────────

def get_db():
    url = os.environ.get('DATABASE_URL')
    if not url:
        raise RuntimeError('DATABASE_URL not set')
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def init_db():
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         SERIAL PRIMARY KEY,
            email      TEXT UNIQUE NOT NULL,
            password   TEXT NOT NULL,
            name       TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS category_master (
            id                       SERIAL PRIMARY KEY,
            user_id                  INTEGER REFERENCES users(id) ON DELETE CASCADE,
            highlevel_classification TEXT,
            code                     TEXT,
            account_name             TEXT NOT NULL,
            type                     TEXT,
            sub_group                TEXT,
            sch_iii_map              TEXT,
            gst_flag                 TEXT,
            tds_section              TEXT,
            boe_flag                 TEXT,
            llm_keywords             TEXT,
            is_active                TEXT DEFAULT 'TRUE',
            repeating_keywords       TEXT,
            if_narration_keyword     TEXT,
            if_not_party_name_keyword TEXT,
            classification           TEXT,
            sort_order               INTEGER DEFAULT 0,
            updated_at               TIMESTAMPTZ DEFAULT NOW()
        );""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS file_cache (
            file_hash     TEXT PRIMARY KEY,
            filename      TEXT NOT NULL,
            user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
            results_json  TEXT NOT NULL,
            stats_json    TEXT NOT NULL,
            classified_at TIMESTAMPTZ DEFAULT NOW()
        );""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS classification_history (
            id         SERIAL PRIMARY KEY,
            user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
            filename   TEXT NOT NULL,
            file_hash  TEXT NOT NULL,
            total_rows INTEGER,
            stats_json TEXT,
            output_csv TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );""")
    conn.commit(); cur.close(); conn.close()
    print("✅ DB tables ready")
    _load_master_into_classifier()


def _load_master_into_classifier(user_id=None):
    """Load category master from DB into the in-memory classifier."""
    try:
        conn = get_db(); cur = conn.cursor()
        if user_id:
            cur.execute("SELECT * FROM category_master WHERE user_id=%s ORDER BY sort_order, id", (user_id,))
        else:
            # Load global master (user_id IS NULL) or the most recently updated user's master
            cur.execute("""
                SELECT * FROM category_master
                WHERE user_id = (
                    SELECT user_id FROM category_master
                    GROUP BY user_id ORDER BY MAX(updated_at) DESC LIMIT 1
                )
                ORDER BY sort_order, id
            """)
        rows = cur.fetchall()
        cur.close(); conn.close()
        if rows:
            classifier.load_from_master_rows([dict(r) for r in rows])
            print(f"✅ Loaded {len(rows)} master rows into classifier")
    except Exception as e:
        print(f"⚠️  Could not load master from DB: {e}")


try:
    init_db()
except Exception as e:
    print(f"⚠️  DB init skipped: {e}")

# ─────────────────────────────────────────────────────────────
#  Auth decorator
# ─────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required', 'auth_required': True}), 401
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def sanitize_value(v):
    if v is None: return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return None if math.isnan(float(v)) else float(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, (np.ndarray,)): return v.tolist()
    if isinstance(v, pd.Timestamp): return str(v)
    return v

def sanitize_records(records):
    return [{k: sanitize_value(v) for k, v in row.items()} for row in records]

def is_category_master_file(df):
    cols_lower = [str(c).lower().strip() for c in df.columns]
    col_str    = ' '.join(cols_lower)
    signals    = ['account name','account_name','sub-group','subgroup','highlevel','high level',
                  'sch iii','schedule','llm keyword','llmkeyword','isactive','is_active',
                  'gst_flag','tds_section','boe_flag','repeating']
    return sum(1 for s in signals if s in col_str) >= 2

def load_df(filepath, filename):
    df = pd.read_csv(filepath, dtype=str) if filename.endswith('.csv') \
         else pd.read_excel(filepath, dtype=str, header=0)
    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
    return df.dropna(how='all').fillna('')

def extract_categories_from_master(df):
    cols_lower  = {str(c).lower().strip(): c for c in df.columns}
    priority    = ['classification','account name','accountname','account_name',
                   'highlevel classification','sub-group','subgroup','category','name']
    target_col  = next((cols_lower[p] for p in priority if p in cols_lower), None)
    if not target_col:
        target_col = next((col for col in df.columns if df[col].dtype == object), None)
    if not target_col: return []
    raw = df[target_col].astype(str).str.strip()
    return sorted({v for v in raw.unique()
                   if v and v.lower() not in ('nan','','none')
                   and not v.replace('.','').replace('-','').isdigit()})

def build_stats(results, confidence_threshold):
    results['Confidence'] = pd.to_numeric(results['Confidence'], errors='coerce').fillna(0.0)
    cat_totals, cat_counts = {}, {}
    amount_col = 'Amount' if 'Amount' in results.columns else None
    for _, row in results.iterrows():
        cat = str(row.get('Category','Other') or 'Other')
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if amount_col:
            try:
                amt = float(str(row[amount_col]).replace(',','').replace('₹','').replace('$','').strip())
                if not math.isnan(amt):
                    cat_totals[cat] = cat_totals.get(cat, 0) + abs(amt)
            except Exception: pass
    base = cat_totals if cat_totals else cat_counts
    sorted_cats = dict(sorted(base.items(), key=lambda x: x[1], reverse=True))
    return {
        'total':              int(len(results)),
        'high_confidence':    int((results['Confidence'] >= confidence_threshold).sum()),
        'needs_review':       int((results['Confidence'] < confidence_threshold).sum()),
        'ai_classified':      int((results['Method'] == 'ai').sum()),
        'keyword_classified': int((results['Method'] == 'keyword').sum()),
        'category_totals':    {str(k): float(v) for k, v in sorted_cats.items()},
        'category_counts':    {str(k): int(v) for k, v in cat_counts.items()},
    }

def _get_current_hash():
    hf = os.path.join(DATA_FOLDER, 'current_hash.txt')
    if os.path.exists(hf):
        parts = open(hf).read().split('|||', 1)
        return parts[0], (parts[1] if len(parts) > 1 else 'transactions.csv')
    return None, 'transactions.csv'

def _persist_results(user_id, fhash, filename, results_safe, stats, output_file):
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            INSERT INTO file_cache (file_hash, filename, user_id, results_json, stats_json)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (file_hash) DO UPDATE
            SET results_json=EXCLUDED.results_json, stats_json=EXCLUDED.stats_json, classified_at=NOW()
        """, (fhash, filename, user_id, json.dumps(results_safe), json.dumps(stats)))
        cur.execute("""
            INSERT INTO classification_history (user_id, filename, file_hash, total_rows, stats_json, output_csv)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (user_id, filename, fhash, stats['total'], json.dumps(stats), output_file))
        conn.commit(); cur.close(); conn.close()
    except Exception as e:
        print(f"⚠️  DB persist failed: {e}")

# ─────────────────────────────────────────────────────────────
#  Auth routes
# ─────────────────────────────────────────────────────────────

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    d = request.get_json() or {}
    name, email, pw = (d.get('name','') or '').strip(), (d.get('email','') or '').strip().lower(), d.get('password','') or ''
    if not name or not email or not pw: return jsonify({'error': 'Name, email and password required'}), 400
    if len(pw) < 6: return jsonify({'error': 'Password must be at least 6 characters'}), 400
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute('SELECT id FROM users WHERE email=%s', (email,))
        if cur.fetchone(): cur.close(); conn.close(); return jsonify({'error': 'Email already registered'}), 409
        cur.execute('INSERT INTO users (email,password,name) VALUES (%s,%s,%s) RETURNING id,name,email',
                    (email, generate_password_hash(pw), name))
        user = dict(cur.fetchone()); conn.commit(); cur.close(); conn.close()
        session.permanent = True
        session.update({'user_id': user['id'], 'user_name': user['name'], 'user_email': user['email']})
        return jsonify({'success': True, 'user': user})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    d = request.get_json() or {}
    email, pw = (d.get('email','') or '').strip().lower(), d.get('password','') or ''
    if not email or not pw: return jsonify({'error': 'Email and password required'}), 400
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute('SELECT id,name,email,password FROM users WHERE email=%s', (email,))
        user = cur.fetchone(); cur.close(); conn.close()
        if not user or not check_password_hash(user['password'], pw):
            return jsonify({'error': 'Invalid email or password'}), 401
        session.permanent = True
        session.update({'user_id': user['id'], 'user_name': user['name'], 'user_email': user['email']})
        # Load this user's master into the classifier
        _load_master_into_classifier(user['id'])
        return jsonify({'success': True, 'user': {'id': user['id'], 'name': user['name'], 'email': user['email']}})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear(); return jsonify({'success': True})

@app.route('/api/auth/me', methods=['GET'])
def me():
    if 'user_id' not in session: return jsonify({'authenticated': False})
    _load_master_into_classifier(session['user_id'])
    return jsonify({'authenticated': True,
                    'user': {'id': session['user_id'], 'name': session['user_name'], 'email': session['user_email']}})

# ─────────────────────────────────────────────────────────────
#  Category Master CRUD
# ─────────────────────────────────────────────────────────────

MASTER_COLS = ['highlevel_classification','code','account_name','type','sub_group','sch_iii_map',
               'gst_flag','tds_section','boe_flag','llm_keywords','is_active',
               'repeating_keywords','if_narration_keyword','if_not_party_name_keyword','classification']

@app.route('/api/master', methods=['GET'])
@login_required
def get_master():
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT * FROM category_master WHERE user_id=%s ORDER BY sort_order, id", (session['user_id'],))
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        return jsonify({'rows': rows, 'count': len(rows)})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/master/row', methods=['POST'])
@login_required
def add_master_row():
    try:
        d = request.get_json() or {}
        if not d.get('account_name'): return jsonify({'error': 'account_name is required'}), 400
        conn = get_db(); cur = conn.cursor()
        cols = ', '.join(MASTER_COLS + ['user_id'])
        vals = ', '.join(['%s'] * (len(MASTER_COLS) + 1))
        data = [d.get(c, '') for c in MASTER_COLS] + [session['user_id']]
        cur.execute(f"INSERT INTO category_master ({cols}) VALUES ({vals}) RETURNING id", data)
        new_id = cur.fetchone()['id']
        conn.commit(); cur.close(); conn.close()
        _load_master_into_classifier(session['user_id'])
        return jsonify({'success': True, 'id': new_id})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/master/row/<int:row_id>', methods=['PUT'])
@login_required
def update_master_row(row_id):
    try:
        d = request.get_json() or {}
        conn = get_db(); cur = conn.cursor()
        # Verify ownership
        cur.execute("SELECT id FROM category_master WHERE id=%s AND user_id=%s", (row_id, session['user_id']))
        if not cur.fetchone(): cur.close(); conn.close(); return jsonify({'error': 'Not found'}), 404
        set_clause = ', '.join([f"{c}=%s" for c in MASTER_COLS]) + ', updated_at=NOW()'
        data = [d.get(c, '') for c in MASTER_COLS] + [row_id, session['user_id']]
        cur.execute(f"UPDATE category_master SET {set_clause} WHERE id=%s AND user_id=%s", data)
        conn.commit(); cur.close(); conn.close()
        _load_master_into_classifier(session['user_id'])
        return jsonify({'success': True})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/master/row/<int:row_id>', methods=['DELETE'])
@login_required
def delete_master_row(row_id):
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("DELETE FROM category_master WHERE id=%s AND user_id=%s", (row_id, session['user_id']))
        conn.commit(); cur.close(); conn.close()
        _load_master_into_classifier(session['user_id'])
        return jsonify({'success': True})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/master/upload', methods=['POST'])
@login_required
def upload_master():
    """Parse uploaded Excel/CSV and upsert all rows into DB."""
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename): return jsonify({'error': 'Invalid file type'}), 400
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        df = load_df(filepath, filename)

        # Normalise column names
        col_map = {
            'highlevel\nclassification': 'highlevel_classification',
            'highlevel classification':  'highlevel_classification',
            'account name':              'account_name',
            'sub-group':                 'sub_group',
            'sch iii map':               'sch_iii_map',
            'gst_flag':                  'gst_flag',
            'tds_section':               'tds_section',
            'boe_flag':                  'boe_flag',
            'llm keywords':              'llm_keywords',
            'isactive':                  'is_active',
            'repeatingkeywords':         'repeating_keywords',
            'repeating keywords':        'repeating_keywords',
            'if(narrationkeyword)':      'if_narration_keyword',
            'ifnot\n(partynamekeyword)': 'if_not_party_name_keyword',
            'ifnot(partynamekeyword)':   'if_not_party_name_keyword',
        }
        df.columns = [col_map.get(c.lower().strip(), c.lower().strip().replace(' ', '_').replace('-', '_'))
                      for c in df.columns]

        conn = get_db(); cur = conn.cursor()
        # Delete existing rows for this user
        cur.execute("DELETE FROM category_master WHERE user_id=%s", (session['user_id'],))

        inserted = 0
        for order, (_, row) in enumerate(df.iterrows()):
            acct = str(row.get('account_name', '') or '').strip()
            if not acct or acct.lower() in ('nan', ''): continue
            data = [row.get(c, '') for c in MASTER_COLS] + [session['user_id'], order]
            cols = ', '.join(MASTER_COLS + ['user_id', 'sort_order'])
            vals = ', '.join(['%s'] * (len(MASTER_COLS) + 2))
            cur.execute(f"INSERT INTO category_master ({cols}) VALUES ({vals})", data)
            inserted += 1

        conn.commit(); cur.close(); conn.close()
        _load_master_into_classifier(session['user_id'])
        return jsonify({'success': True, 'inserted': inserted,
                        'message': f'Loaded {inserted} accounts into your category master.'})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/master/export', methods=['GET'])
@login_required
def export_master():
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT * FROM category_master WHERE user_id=%s ORDER BY sort_order, id", (session['user_id'],))
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        if not rows: return jsonify({'error': 'No master data to export'}), 404
        df = pd.DataFrame(rows).drop(columns=['id','user_id','updated_at'], errors='ignore')
        out = os.path.join(OUTPUT_FOLDER, 'category_master_export.xlsx')
        df.to_excel(out, index=False)
        return send_file(out, as_attachment=True, download_name='category_master.xlsx')
    except Exception as e: return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────
#  History routes
# ─────────────────────────────────────────────────────────────

@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("""SELECT id,filename,file_hash,total_rows,stats_json,output_csv,created_at
                       FROM classification_history WHERE user_id=%s
                       ORDER BY created_at DESC LIMIT 50""", (session['user_id'],))
        rows = cur.fetchall(); cur.close(); conn.close()
        history = []
        for r in rows:
            e = dict(r); e['stats'] = json.loads(e.pop('stats_json') or '{}')
            e['created_at'] = str(e['created_at']); history.append(e)
        return jsonify({'history': history})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/history/<int:hid>/results', methods=['GET'])
@login_required
def get_history_results(hid):
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("""SELECT h.file_hash,h.filename,fc.results_json,fc.stats_json
                       FROM classification_history h
                       JOIN file_cache fc ON fc.file_hash=h.file_hash
                       WHERE h.id=%s AND h.user_id=%s""", (hid, session['user_id']))
        row = cur.fetchone(); cur.close(); conn.close()
        if not row: return jsonify({'error': 'Not found'}), 404
        return jsonify({'success': True, 'results': json.loads(row['results_json']),
                        'stats': json.loads(row['stats_json']), 'filename': row['filename']})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/history/<int:hid>/download', methods=['GET'])
@login_required
def download_history_file(hid):
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT output_csv,filename FROM classification_history WHERE id=%s AND user_id=%s",
                    (hid, session['user_id']))
        row = cur.fetchone(); cur.close(); conn.close()
        if not row or not row['output_csv'] or not os.path.exists(row['output_csv']):
            return jsonify({'error': 'File not found'}), 404
        return send_file(row['output_csv'], mimetype='text/csv', as_attachment=True,
                         download_name=f"classified_{row['filename']}")
    except Exception as e: return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────
#  Upload & classify
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files: return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not file or file.filename == '': return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename): return jsonify({'error': 'Invalid file type'}), 400
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        fhash = compute_file_hash(filepath)

        # Check cache
        already = False
        try:
            conn = get_db(); cur = conn.cursor()
            cur.execute('SELECT 1 FROM file_cache WHERE file_hash=%s AND user_id=%s', (fhash, session['user_id']))
            already = cur.fetchone() is not None; cur.close(); conn.close()
        except Exception: pass

        df = load_df(filepath, filename)

        if is_category_master_file(df):
            # Route through the dedicated master upload endpoint logic
            col_map = {
                'highlevel\nclassification': 'highlevel_classification',
                'highlevel classification':  'highlevel_classification',
                'account name':  'account_name',
                'sub-group':     'sub_group',
                'sch iii map':   'sch_iii_map',
                'llm keywords':  'llm_keywords',
                'isactive':      'is_active',
                'repeatingkeywords': 'repeating_keywords',
                'repeating keywords': 'repeating_keywords',
                'if(narrationkeyword)': 'if_narration_keyword',
                'ifnot\n(partynamekeyword)': 'if_not_party_name_keyword',
                'ifnot(partynamekeyword)': 'if_not_party_name_keyword',
            }
            df2 = df.copy()
            df2.columns = [col_map.get(c.lower().strip(), c.lower().strip().replace(' ','_').replace('-','_'))
                           for c in df2.columns]
            conn = get_db(); cur = conn.cursor()
            cur.execute("DELETE FROM category_master WHERE user_id=%s", (session['user_id'],))
            inserted = 0
            for order, (_, row) in enumerate(df2.iterrows()):
                acct = str(row.get('account_name','') or '').strip()
                if not acct or acct.lower() in ('nan',''): continue
                data = [row.get(c,'') for c in MASTER_COLS] + [session['user_id'], order]
                cols = ', '.join(MASTER_COLS + ['user_id','sort_order'])
                vals = ', '.join(['%s'] * (len(MASTER_COLS) + 2))
                cur.execute(f"INSERT INTO category_master ({cols}) VALUES ({vals})", data)
                inserted += 1
            conn.commit(); cur.close(); conn.close()
            _load_master_into_classifier(session['user_id'])
            preview = sanitize_records(df.head(8).to_dict('records'))
            return jsonify({'success': True, 'file_mode': 'category_master', 'filename': filename,
                            'total_rows': len(df), 'columns': list(df.columns), 'preview': preview,
                            'categories_loaded': inserted, 'categories_sample': [],
                            'message': f'{inserted} accounts saved to your category master.'})

        # Transactions
        df.to_csv(os.path.join(DATA_FOLDER, 'current_transactions.csv'), index=False)
        with open(os.path.join(DATA_FOLDER, 'current_hash.txt'), 'w') as f:
            f.write(f'{fhash}|||{filename}')

        file_type = classifier.detect_file_type(df)
        return jsonify({'success': True, 'file_mode': 'transactions', 'filename': filename,
                        'file_hash': fhash, 'total_rows': len(df), 'columns': list(df.columns),
                        'preview': sanitize_records(df.head(5).to_dict('records')),
                        'file_type': file_type,
                        'suggested_categories': classifier.get_auto_categories(file_type),
                        'has_category_master': classifier.categories_loaded,
                        'already_classified': already,
                        'message': 'This file was already classified. Load previous results or re-classify.' if already else ''})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500

@app.route('/api/load-cached', methods=['POST'])
@login_required
def load_cached():
    try:
        hf = os.path.join(DATA_FOLDER, 'current_hash.txt')
        if not os.path.exists(hf): return jsonify({'error': 'No file uploaded yet'}), 400
        fhash, filename = open(hf).read().split('|||', 1)
        conn = get_db(); cur = conn.cursor()
        cur.execute('SELECT results_json,stats_json FROM file_cache WHERE file_hash=%s AND user_id=%s',
                    (fhash, session['user_id']))
        cached = cur.fetchone(); cur.close(); conn.close()
        if not cached: return jsonify({'error': 'No cached results found'}), 404
        return jsonify({'success': True, 'results': json.loads(cached['results_json']),
                        'stats': json.loads(cached['stats_json']), 'from_cache': True})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/force-transactions', methods=['POST'])
@login_required
def force_transactions():
    try:
        mf = os.path.join(DATA_FOLDER, 'category_master.csv')
        if not os.path.exists(mf): return jsonify({'error': 'No master file found'}), 400
        df = pd.read_csv(mf, dtype=str).fillna('')
        df.to_csv(os.path.join(DATA_FOLDER, 'current_transactions.csv'), index=False)
        return jsonify({'success': True, 'total_rows': len(df), 'file_type': classifier.detect_file_type(df),
                        'message': f'File re-interpreted as transactions ({len(df)} rows)'})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/classify-stream', methods=['GET'])
@login_required
def classify_stream():
    import queue, threading
    use_ai = request.args.get('use_ai','true').lower() == 'true'
    conf_threshold = float(request.args.get('confidence_threshold', 0.6))
    raw_cats = request.args.get('categories', None)
    custom_cats = json.loads(raw_cats) if raw_cats else None
    user_id = session['user_id']
    fhash, fname = _get_current_hash()

    trans_file = os.path.join(DATA_FOLDER, 'current_transactions.csv')
    if not os.path.exists(trans_file):
        def err_gen():
            yield 'data: ' + json.dumps({'error': 'No transactions file found'}) + '\n\n'
        return app.response_class(err_gen(), mimetype='text/event-stream')

    df = pd.read_csv(trans_file, dtype=str).fillna('')
    q  = queue.Queue()

    def worker():
        try:
            results, file_type, cats_used = classifier.classify_transactions(
                df, categories=custom_cats, use_ai=use_ai,
                confidence_threshold=conf_threshold, progress_queue=q)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(OUTPUT_FOLDER, f'classified_{ts}.csv')
            results.to_csv(out, index=False)
            stats = build_stats(results, conf_threshold)
            stats['file_type'] = str(file_type)
            stats['categories_used'] = [str(c) for c in cats_used]
            safe = sanitize_records(results.where(pd.notnull(results), None).to_dict('records'))
            if fhash: _persist_results(user_id, fhash, fname, safe, stats, out)
            q.put({'type': 'done', 'results': safe, 'stats': stats, 'output_file': str(out)})
        except Exception as e:
            import traceback; traceback.print_exc()
            q.put({'type': 'error', 'error': str(e)})

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        yield 'data: ' + json.dumps({'type': 'start', 'total': len(df)}) + '\n\n'
        while True:
            try: msg = q.get(timeout=120)
            except Exception:
                yield 'data: ' + json.dumps({'type': 'error', 'error': 'Timeout'}) + '\n\n'; return
            yield 'data: ' + json.dumps(msg) + '\n\n'
            if msg.get('type') in ('done', 'error'): return

    return app.response_class(generate(), mimetype='text/event-stream',
                               headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/classify', methods=['POST'])
@login_required
def classify_transactions_route():
    try:
        tf = os.path.join(DATA_FOLDER, 'current_transactions.csv')
        if not os.path.exists(tf): return jsonify({'error': 'Upload a transactions file first'}), 400
        df = pd.read_csv(tf, dtype=str).fillna('')
        d  = request.get_json() or {}
        ct = float(d.get('confidence_threshold', 0.6))
        results, ft, cats = classifier.classify_transactions(
            df, categories=d.get('categories'), use_ai=d.get('use_ai', True), confidence_threshold=ct)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(OUTPUT_FOLDER, f'classified_{ts}.csv')
        results.to_csv(out, index=False)
        stats = build_stats(results, ct)
        stats.update({'file_type': str(ft), 'categories_used': [str(c) for c in cats]})
        safe = sanitize_records(results.where(pd.notnull(results), None).to_dict('records'))
        fhash, fname = _get_current_hash()
        if fhash: _persist_results(session['user_id'], fhash, fname, safe, stats, out)
        return jsonify({'success': True, 'results': safe, 'stats': stats, 'output_file': str(out)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Classification error: {str(e)}'}), 500

@app.route('/api/update-classification', methods=['POST'])
@login_required
def update_classification():
    try:
        d = request.get_json()
        if not d.get('description') or not d.get('category'):
            return jsonify({'error': 'Missing description or category'}), 400
        classifier.learn_mapping(d['description'], d['category'])
        return jsonify({'success': True})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/download-results', methods=['GET'])
@login_required
def download_results():
    try:
        files = [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith('classified_')]
        if not files: return jsonify({'error': 'No results available'}), 404
        fp = os.path.join(OUTPUT_FOLDER, max(files))
        return send_file(fp, mimetype='text/csv', as_attachment=True, download_name='classified_transactions.csv')
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        auth = 'user_id' in session
        return jsonify({
            'learned_mappings':  int(len(classifier.vendor_memory)),
            'ai_provider':       str(classifier.ai_provider or 'none'),
            'ai_model':          str(getattr(classifier, 'openrouter_model', 'n/a')),
            'categories_loaded': bool(classifier.categories_loaded),
            'total_categories':  int(len(classifier.categories)) if classifier.categories_loaded else 0,
            'authenticated':     auth,
            'user_name':         session.get('user_name') if auth else None,
            'user_email':        session.get('user_email') if auth else None,
        })
    except Exception as e: return jsonify({'error': str(e)}), 500

# Timing endpoints
_tsf = os.path.join(DATA_FOLDER, 'timing_store.json')
def _lt():
    try: return json.load(open(_tsf)) if os.path.exists(_tsf) else {'ai':[],'keyword':[]}
    except: return {'ai':[],'keyword':[]}
def _st(d): json.dump(d, open(_tsf,'w'))

@app.route('/api/timing', methods=['GET'])
def get_timing():
    s = _lt()
    return jsonify({m: round(sorted(s.get(m,[])[-5:])[len(s.get(m,[])[-5:])//2],4)
                    if s.get(m) else (0.9 if m=='ai' else 0.08) for m in ('ai','keyword')})

@app.route('/api/timing', methods=['POST'])
def post_timing():
    try:
        d = request.get_json() or {}
        rows, elapsed, mode = int(d.get('rows',0)), float(d.get('elapsed_sec',0)), d.get('mode','ai')
        if rows<=0 or elapsed<=0: return jsonify({'ok':False}), 400
        s = _lt(); s.setdefault(mode,[]).append(round(elapsed/rows,4)); s[mode]=s[mode][-20:]; _st(s)
        return jsonify({'ok':True})
    except Exception as e: return jsonify({'error':str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port)