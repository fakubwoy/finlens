from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import math
from datetime import datetime
from classifier import ExpenseClassifier
from werkzeug.utils import secure_filename

# Load .env file if present (never hard-code secrets)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — env vars must be set externally

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
DATA_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, DATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

classifier = ExpenseClassifier()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def sanitize_value(v):
    """Convert any non-JSON-safe value to a safe Python type."""
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
    """Sanitize a list of dicts for JSON serialization."""
    clean = []
    for row in records:
        clean.append({k: sanitize_value(v) for k, v in row.items()})
    return clean


def is_category_master_file(df):
    """
    Detect if the uploaded file is a chart of accounts / category master
    rather than a transactions file.
    """
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
    """Read CSV or Excel cleanly, handling merged cells and multi-line headers."""
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath, dtype=str)
    else:
        df = pd.read_excel(filepath, dtype=str, header=0)

    # Clean column names
    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
    df = df.dropna(how='all')
    df = df.fillna('')
    return df


def extract_categories_from_master(df):
    """Pull meaningful category/account names from a chart of accounts."""
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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload CSV or Excel (.xlsx, .xls)'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = load_df(filepath, filename)

        if is_category_master_file(df):
            # Chart of accounts / category master
            df.to_csv(os.path.join(DATA_FOLDER, 'category_master.csv'), index=False)
            categories = extract_categories_from_master(df)
            classifier.load_categories_from_master(df, categories)

            preview = sanitize_records(df.head(8).to_dict('records'))
            columns = list(df.columns)

            return jsonify({
                'success': True,
                'file_mode': 'category_master',
                'filename': filename,
                'total_rows': len(df),
                'columns': columns,
                'preview': preview,
                'categories_loaded': len(categories),
                'categories_sample': categories[:20],
                'message': f'Category master loaded — {len(categories)} categories found. Now upload your transactions file.'
            })

        else:
            # Transactions file
            df.to_csv(os.path.join(DATA_FOLDER, 'current_transactions.csv'), index=False)
            file_type = classifier.detect_file_type(df)
            suggested_categories = classifier.get_auto_categories(file_type)

            preview = sanitize_records(df.head(5).to_dict('records'))
            columns = list(df.columns)

            return jsonify({
                'success': True,
                'file_mode': 'transactions',
                'filename': filename,
                'total_rows': len(df),
                'columns': columns,
                'preview': preview,
                'file_type': file_type,
                'suggested_categories': suggested_categories,
                'has_category_master': classifier.categories_loaded
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500


@app.route('/api/force-transactions', methods=['POST'])
def force_transactions():
    """Re-interpret the last uploaded master file as a transactions file."""
    try:
        master_file = os.path.join(DATA_FOLDER, 'category_master.csv')
        if not os.path.exists(master_file):
            return jsonify({'error': 'No master file found to convert'}), 400

        df = pd.read_csv(master_file, dtype=str).fillna('')
        # Save it as the transactions file
        trans_path = os.path.join(DATA_FOLDER, 'current_transactions.csv')
        df.to_csv(trans_path, index=False)

        file_type = classifier.detect_file_type(df)
        return jsonify({
            'success': True,
            'total_rows': len(df),
            'file_type': file_type,
            'message': f'File re-interpreted as transactions ({len(df)} rows)'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classify-stream', methods=['GET'])
def classify_stream():
    """
    SSE endpoint that streams real per-row progress during classification.
    Query params mirror the JSON body of /api/classify:
      use_ai, confidence_threshold, categories (JSON array)
    """
    import queue, threading

    use_ai = request.args.get('use_ai', 'true').lower() == 'true'
    confidence_threshold = float(request.args.get('confidence_threshold', 0.6))
    raw_cats = request.args.get('categories', None)
    custom_categories = json.loads(raw_cats) if raw_cats else None

    trans_file = os.path.join(DATA_FOLDER, 'current_transactions.csv')
    if not os.path.exists(trans_file):
        def err_gen():
            yield 'data: ' + json.dumps({'error': 'No transactions file found'}) + '\n\n'
        return app.response_class(err_gen(), mimetype='text/event-stream')

    df = pd.read_csv(trans_file, dtype=str).fillna('')
    total_rows = len(df)

    # Queue carries progress dicts from the worker thread to the generator
    q = queue.Queue()

    def worker():
        try:
            results, file_type, categories_used = classifier.classify_transactions(
                df,
                categories=custom_categories,
                use_ai=use_ai,
                confidence_threshold=confidence_threshold,
                progress_queue=q          # new kwarg — see classifier.py
            )

            # Save output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(OUTPUT_FOLDER, f'classified_{timestamp}.csv')
            results.to_csv(output_file, index=False)

            # Build stats (same logic as /api/classify)
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

            stats = {
                'total': int(len(results)),
                'high_confidence': int((results['Confidence'] >= confidence_threshold).sum()),
                'needs_review': int((results['Confidence'] < confidence_threshold).sum()),
                'ai_classified': int((results['Method'] == 'ai').sum()),
                'keyword_classified': int((results['Method'] == 'keyword').sum()),
                'category_totals': {str(k): float(v) for k, v in sorted_cats.items()},
                'category_counts': {str(k): int(v) for k, v in category_counts.items()},
                'file_type': str(file_type),
                'categories_used': [str(c) for c in categories_used]
            }

            results_safe = sanitize_records(
                results.where(pd.notnull(results), None).to_dict('records')
            )

            q.put({'type': 'done', 'results': results_safe, 'stats': stats, 'output_file': str(output_file)})
        except Exception as e:
            import traceback; traceback.print_exc()
            q.put({'type': 'error', 'error': str(e)})

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    def generate():
        # Send total first so frontend knows the denominator immediately
        yield 'data: ' + json.dumps({'type': 'start', 'total': total_rows}) + '\n\n'
        while True:
            try:
                msg = q.get(timeout=120)   # 2-min hard timeout per message
            except Exception:
                yield 'data: ' + json.dumps({'type': 'error', 'error': 'Timeout waiting for classifier'}) + '\n\n'
                return
            yield 'data: ' + json.dumps(msg) + '\n\n'
            if msg.get('type') in ('done', 'error'):
                return

    return app.response_class(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',   # disable nginx buffering if behind proxy
        }
    )


@app.route('/api/classify', methods=['POST'])
def classify_transactions():
    try:
        trans_file = os.path.join(DATA_FOLDER, 'current_transactions.csv')
        if not os.path.exists(trans_file):
            return jsonify({'error': 'Please upload a transactions file first'}), 400

        df = pd.read_csv(trans_file, dtype=str).fillna('')

        req_data = request.get_json() or {}
        confidence_threshold = float(req_data.get('confidence_threshold', 0.6))
        use_ai = req_data.get('use_ai', True)
        custom_categories = req_data.get('categories', None)

        results, file_type, categories_used = classifier.classify_transactions(
            df,
            categories=custom_categories,
            use_ai=use_ai,
            confidence_threshold=confidence_threshold
        )

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_FOLDER, f'classified_{timestamp}.csv')
        results.to_csv(output_file, index=False)

        # Coerce Confidence to numeric safely
        results['Confidence'] = pd.to_numeric(results['Confidence'], errors='coerce').fillna(0.0)

        category_totals = {}
        category_counts = {}
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

        stats = {
            'total': int(len(results)),
            'high_confidence': int((results['Confidence'] >= confidence_threshold).sum()),
            'needs_review': int((results['Confidence'] < confidence_threshold).sum()),
            'ai_classified': int((results['Method'] == 'ai').sum()),
            'keyword_classified': int((results['Method'] == 'keyword').sum()),
            'category_totals': {str(k): float(v) for k, v in sorted_cats.items()},
            'category_counts': {str(k): int(v) for k, v in category_counts.items()},
            'file_type': str(file_type),
            'categories_used': [str(c) for c in categories_used]
        }

        results_safe = sanitize_records(
            results.where(pd.notnull(results), None).to_dict('records')
        )

        return jsonify({
            'success': True,
            'results': results_safe,
            'stats': stats,
            'output_file': str(output_file)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Classification error: {str(e)}'}), 500


@app.route('/api/update-classification', methods=['POST'])
def update_classification():
    try:
        data = request.get_json()
        description = data.get('description')
        new_category = data.get('category')
        if not description or not new_category:
            return jsonify({'error': 'Missing description or category'}), 400
        classifier.learn_mapping(description, new_category)
        return jsonify({'success': True, 'message': f'Learned: "{description}" → {new_category}'})
    except Exception as e:
        return jsonify({'error': f'Update error: {str(e)}'}), 500


@app.route('/api/download-results', methods=['GET'])
def download_results():
    try:
        output_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith('classified_')]
        if not output_files:
            return jsonify({'error': 'No results available'}), 404
        latest_file = max(output_files)
        filepath = os.path.join(OUTPUT_FOLDER, latest_file)
        return send_file(filepath, mimetype='text/csv', as_attachment=True,
                         download_name='classified_transactions.csv')
    except Exception as e:
        return jsonify({'error': f'Download error: {str(e)}'}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        return jsonify({
            'learned_mappings': int(len(classifier.vendor_memory)),
            'ai_provider': str(classifier.ai_provider or 'none'),
            'ai_model': str(getattr(classifier, 'openrouter_model', 'n/a')),
            'categories_loaded': bool(classifier.categories_loaded),
            'total_categories': int(len(classifier.categories)) if classifier.categories_loaded else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Timing telemetry — frontend POSTs real elapsed + row count after each run
# so the next estimate is evidence-based rather than hard-coded.
# ---------------------------------------------------------------------------
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
    """Return per-row speed estimates (seconds/row) for AI and keyword modes."""
    store = _load_timing()
    result = {}
    for mode in ('ai', 'keyword'):
        samples = store.get(mode, [])
        if samples:
            # Use median of last 5 samples to smooth outliers
            recent = samples[-5:]
            median_spr = sorted(recent)[len(recent) // 2]
            result[mode] = round(median_spr, 4)
        else:
            result[mode] = 0.9 if mode == 'ai' else 0.08  # sensible defaults
    return jsonify(result)

@app.route('/api/timing', methods=['POST'])
def post_timing():
    """Record actual elapsed time for a classification run."""
    try:
        data = request.get_json() or {}
        rows = int(data.get('rows', 0))
        elapsed_sec = float(data.get('elapsed_sec', 0))
        mode = data.get('mode', 'ai')   # 'ai' or 'keyword'
        if rows <= 0 or elapsed_sec <= 0:
            return jsonify({'ok': False, 'reason': 'invalid data'}), 400
        spr = elapsed_sec / rows
        store = _load_timing()
        store.setdefault(mode, []).append(round(spr, 4))
        store[mode] = store[mode][-20:]   # keep last 20 observations
        _save_timing(store)
        return jsonify({'ok': True, 'sec_per_row': spr})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Smart Expense Classifier Starting on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port)