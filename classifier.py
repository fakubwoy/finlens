import pandas as pd
import re
import json
import os
import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional


class ExpenseClassifier:
    def __init__(self):
        self.categories = []           # list of full row dicts from master
        self.category_keywords = {}    # account_name -> [keywords]
        self.category_meta = {}        # account_name -> {code, gst_flag, tds_section, boe_flag, ...}
        self.conditional_rules = []    # [{account_name, if_narration, if_not_party}]
        self.vendor_memory = {}
        self.categories_loaded = False
        self.memory_file = 'data/vendor_memory.json'
        self.client = None
        self.ai_provider = None

        self.load_vendor_memory()
        self._init_ai()

    def _init_ai(self):
        or_key = os.environ.get('OPENROUTER_API_KEY')
        if not or_key:
            print("⚠️  [AI] OPENROUTER_API_KEY not set — keyword-only mode active")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=or_key,
                base_url='https://openrouter.ai/api/v1',
                default_headers={
                    'HTTP-Referer': 'https://expense-classifier.local',
                    'X-Title': 'Indian Expense Classifier',
                },
            )
            self.openrouter_model = os.environ.get(
                'OPENROUTER_MODEL', 'meta-llama/llama-3.3-70b-instruct'
            )
            self.ai_provider = 'openrouter'
            print(f"✅ [AI] OpenRouter initialised | model: {self.openrouter_model}")
        except ImportError:
            print("⚠️  [AI] openai package not installed")
        except Exception as e:
            print(f"⚠️  [AI] OpenRouter init failed: {e}")

    def load_vendor_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.vendor_memory = json.load(f)
            except Exception:
                self.vendor_memory = {}

    def save_vendor_memory(self):
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, 'w') as f:
            json.dump(self.vendor_memory, f, indent=2)

    # ─────────────────────────────────────────────────────────
    #  Load from master (DB rows or DataFrame)
    # ─────────────────────────────────────────────────────────

    def load_from_master_rows(self, rows: List[dict]):
        """
        Load category master from a list of dicts (from DB or parsed Excel).
        Expected keys (case-insensitive): account_name, code, llm_keywords,
        repeating_keywords, gst_flag, tds_section, boe_flag,
        if_narration_keyword, if_not_party_name_keyword, is_active
        """
        self.categories = []
        self.category_keywords = {}
        self.category_meta = {}
        self.conditional_rules = []

        def _get(row, *keys):
            for k in keys:
                v = row.get(k) or row.get(k.lower()) or row.get(k.upper())
                if v is not None and str(v).strip() not in ('', 'nan', 'None'):
                    return str(v).strip()
            return ''

        for row in rows:
            name = _get(row, 'account_name', 'Account Name', 'AccountName')
            if not name:
                continue

            is_active = _get(row, 'is_active', 'IsActive', 'isactive')
            if is_active.lower() == 'false':
                continue

            code         = _get(row, 'code', 'Code')
            gst_flag     = _get(row, 'gst_flag', 'GST_Flag', 'gstflag')
            tds_section  = _get(row, 'tds_section', 'TDS_Section', 'tdssection')
            boe_flag     = _get(row, 'boe_flag', 'BOE_FLAG', 'boeflag')
            highlevel    = _get(row, 'highlevel_classification', 'HighLevel Classification',
                                 'HighLevel\nClassification', 'highlevel')
            sub_group    = _get(row, 'sub_group', 'Sub-Group', 'SubGroup', 'subgroup')
            acct_type    = _get(row, 'type', 'Type')

            self.category_meta[name] = {
                'code': code,
                'gst_flag': gst_flag,
                'tds_section': tds_section,
                'boe_flag': boe_flag,
                'highlevel': highlevel,
                'sub_group': sub_group,
                'type': acct_type,
            }
            self.categories.append({'Category': name, **self.category_meta[name]})

            # Collect keywords from llm_keywords + repeating_keywords
            kws = set()
            for kf in ('llm_keywords', 'LLM Keywords', 'llmkeywords',
                       'repeating_keywords', 'repeatingKeywords', 'repeatingkeywords'):
                raw = _get(row, kf)
                if raw:
                    for k in re.split(r'[,;]', raw):
                        k = k.strip().lower()
                        if k:
                            kws.add(k)
            self.category_keywords[name] = list(kws)

            # Conditional rules
            if_narr     = _get(row, 'if_narration_keyword', 'if(narrationKeyword)',
                                'ifnarrationkeyword', 'if_narration')
            if_not_party = _get(row, 'if_not_party_name_keyword', 'ifNot(partyNameKeyword)',
                                 'ifnotpartynamekeyword', 'if_not_party')
            if if_narr or if_not_party:
                self.conditional_rules.append({
                    'account_name': name,
                    'if_narration': [k.strip().lower() for k in re.split(r'[,;]', if_narr) if k.strip()] if if_narr else [],
                    'if_not_party': [k.strip().lower() for k in re.split(r'[,;]', if_not_party) if k.strip()] if if_not_party else [],
                })

        self.categories_loaded = bool(self.categories)
        total_kws = sum(len(v) for v in self.category_keywords.values())
        print(f"✅ Loaded {len(self.categories)} categories, {total_kws} keywords, "
              f"{len(self.conditional_rules)} conditional rules")

    def load_categories_from_master(self, df: pd.DataFrame, category_names: List[str]):
        """Legacy shim — convert DataFrame to row dicts and call load_from_master_rows."""
        rows = df.to_dict('records')
        # Normalise column names to snake_case keys the loader understands
        norm_rows = []
        for row in rows:
            norm = {}
            for k, v in row.items():
                nk = str(k).lower().replace(' ', '_').replace('-', '_').replace('\n', '_')
                norm[nk] = v
            norm_rows.append(norm)
        self.load_from_master_rows(norm_rows)

    def load_categories(self, df: pd.DataFrame):
        """Simple category/keywords CSV — kept for backwards compat."""
        rows = []
        for _, r in df.iterrows():
            rows.append({
                'account_name': str(r.get('Category', '') or ''),
                'llm_keywords': str(r.get('Keywords', '') or ''),
            })
        self.load_from_master_rows(rows)

    # ─────────────────────────────────────────────────────────
    #  Classification helpers
    # ─────────────────────────────────────────────────────────

    def detect_file_type(self, df: pd.DataFrame) -> str:
        cols = ' '.join(c.lower() for c in df.columns)
        if any(x in cols for x in ['debit', 'credit', 'balance', 'withdrawal', 'deposit']):
            return 'bank_statement'
        if any(x in cols for x in ['expense', 'reimburs', 'receipt']):
            return 'expense_report'
        if any(x in cols for x in ['invoice', 'vendor', 'supplier', 'purchase order', 'po ']):
            return 'business_transactions'
        return 'financial_data'

    def get_auto_categories(self, file_type: str) -> List[str]:
        if file_type == 'bank_statement':
            return [
                'Salary & Wages', 'Business Income', 'Interest Income', 'Dividend Income',
                'Bank Transfer (NEFT/IMPS/RTGS)', 'UPI Transfer', 'ATM Withdrawal', 'Cheque Deposit',
                'Food & Dining', 'Groceries & Kirana', 'Shopping & Retail',
                'Transport & Commute', 'Fuel & Petrol',
                'Rent & Housing', 'Electricity & Power', 'Water & Municipal',
                'Mobile & Internet Recharge', 'DTH & Cable TV',
                'EMI & Loan Repayment', 'Credit Card Payment', 'Insurance Premium',
                'Mutual Fund / SIP', 'Investment & Shares', 'Fixed Deposit',
                'Income Tax (TDS/Advance)', 'GST Payment', 'Professional Tax',
                'PF / ESI Contribution', 'Govt Fees & Challan',
                'Healthcare & Medical', 'Education & Coaching', 'Entertainment & OTT',
                'Travel & Hotel', 'Personal Care & Salon', 'Other',
            ]
        elif file_type == 'business_transactions':
            return [
                'Sales / Revenue', 'Service Income', 'Other Operating Income',
                'Purchase of Stock-in-Trade', 'Raw Material Purchase', 'Freight Inward',
                'Salaries & Wages', 'Director Remuneration', 'ESI & PF Contribution',
                'Staff Welfare', 'Bonus & Ex-Gratia',
                'Rent & Lease', 'Office Supplies & Stationery', 'Telephone & Internet',
                'Electricity Charges', 'Repairs & Maintenance',
                'Bank Charges & Commission', 'Loan Interest', 'EMI & Loan Repayment',
                'GST Payment', 'TDS / TCS Deposit', 'Income Tax Payment',
                'Professional Tax', 'PF / ESI Deposit',
                'Advertising & Marketing', 'Business Promotion', 'Commission & Brokerage',
                'Professional & Legal Fees', 'Audit & Accounting Fees',
                'Software & SaaS Subscriptions', 'Cloud Infrastructure',
                'IT & Hardware Purchase', 'Capital Expenditure',
                'Travel & Conveyance', 'Hotel & Accommodation', 'Meals & Entertainment',
                'Insurance Premium', 'Bank Transfer', 'Miscellaneous Expense', 'Other',
            ]
        return [
            'Food & Dining', 'Groceries', 'Transport', 'Rent & Housing',
            'Electricity & Utilities', 'Mobile Recharge', 'Healthcare',
            'Education', 'Entertainment', 'Insurance',
            'EMI & Loan', 'Investment / SIP', 'Tax Payment',
            'Bank Transfer', 'ATM & Cash', 'Other'
        ]

    def preprocess_description(self, description: str) -> str:
        if not description or str(description).lower() in ('nan', 'none', ''):
            return ""
        desc = str(description).lower()
        desc = re.sub(
            r'^(upi|neft|imps|rtgs|pos|atm|trf|ach|wire|nach|si|enach|'
            r'me dc si|billdesk|razorpay\*|payu\*|ccavenue\*|cashfree\*|'
            r'paytm\*|phonepe\*|gpay\*|googlepay\*|mobikwik\*|freecharge\*|'
            r'iob|sbi|hdfc|icici|axis|kotak|yes bank|bob|pnb|'
            r'cdm deposit|cash dep|chq dep|cheque dep)\s*[-:/\*]?\s*',
            '', desc
        )
        desc = re.sub(r'\b[a-z0-9]{12,}\b', '', desc)
        desc = re.sub(r'\b\d{6,}\b', '', desc)
        desc = re.sub(r'[^a-z0-9\s&]', ' ', desc)
        desc = re.sub(r'\s+', ' ', desc).strip()
        return desc

    def extract_vendor(self, description: str) -> str:
        clean = self.preprocess_description(description)
        words = clean.split()
        return ' '.join(words[:min(3, len(words))]) if words else clean

    def keyword_match(self, description: str, categories: List[str]) -> Tuple[Optional[str], float]:
        clean_desc = self.preprocess_description(description)
        raw_lower  = description.lower()

        vendor = self.extract_vendor(description)
        if vendor and vendor in self.vendor_memory:
            mem_cat = self.vendor_memory[vendor]
            if mem_cat in categories:
                return mem_cat, 0.95

        # ── 1. Conditional rules (if_narration + if_not_party) ──
        for rule in self.conditional_rules:
            name = rule['account_name']
            if name not in categories:
                continue
            if_narr     = rule.get('if_narration', [])
            if_not_party = rule.get('if_not_party', [])

            if if_narr:
                narr_hit = any(kw in raw_lower for kw in if_narr)
                if not narr_hit:
                    continue
            if if_not_party:
                party_hit = any(kw in raw_lower for kw in if_not_party)
                if party_hit:
                    continue
            return name, 0.92

        # ── 2. Standard keyword match (master keywords) ──
        best_match = None
        best_score = 0.0

        for category, keywords in self.category_keywords.items():
            if category not in categories:
                continue
            for keyword in keywords:
                if keyword and keyword in clean_desc:
                    score = min(0.85 + (len(keyword) / max(len(clean_desc), 1)) * 0.1, 0.94)
                    if score > best_score:
                        best_score = score
                        best_match = category

        if best_match:
            return best_match, best_score

        # ── 3. Built-in fallback rules ──
        builtin_rules = {
            'Food & Dining': ['swiggy', 'zomato', 'restaurant', 'cafe', 'food', 'pizza', 'burger',
                              'bakery', 'starbucks', 'mcdonalds', 'kfc', 'dominos', 'canteen',
                              'dunzo', 'zepto food', 'blinkit food', 'haldiram', 'subway'],
            'Shopping & Retail': ['amazon', 'flipkart', 'myntra', 'nykaa', 'ajio', 'shop', 'store',
                                  'mart', 'retail', 'mall', 'bazaar', 'bigbasket', 'blinkit', 'zepto',
                                  'vijay sales', 'croma', 'reliance digital', 'meesho', 'snapdeal',
                                  'tatacliq', 'd mart', 'dmart', 'jiomart'],
            'Transport & Travel': ['uber', 'ola', 'rapido', 'redbus', 'irctc', 'airline', 'flight',
                                   'train', 'bus', 'taxi', 'cab', 'metro', 'petrol', 'fuel',
                                   'makemytrip', 'easytrip', 'easymytrip', 'indigo', 'spicejet',
                                   'goibibo', 'indian railways', 'namma metro'],
            'Utilities & Bills': ['electricity', 'water', 'gas', 'broadband', 'airtel', 'jio', 'bsnl',
                                  'vodafone', 'internet', 'mobile', 'recharge', 'bill', 'tata power',
                                  'bescom', 'msedcl', 'tneb', 'act fibernet', 'hathway'],
            'Healthcare & Medical': ['hospital', 'clinic', 'pharmacy', 'medical', 'doctor', 'health',
                                     'medicine', 'lab', 'diagnostic', 'apollo', 'fortis', 'dental',
                                     'medplus', 'netmeds', 'pharmeasy', '1mg', 'eye hospital'],
            'Entertainment': ['netflix', 'amazon prime', 'hotstar', 'spotify', 'youtube', 'cinema',
                               'movie', 'pvr', 'inox', 'bookmyshow', 'disney', 'zee5', 'sonyliv'],
            'Software & Subscriptions': ['aws', 'azure', 'google cloud', 'github', 'slack', 'zoom',
                                         'microsoft', 'adobe', 'chatgpt', 'openai', 'googleworksp',
                                         'google workspace', 'dropbox', 'notion', 'figma', 'canva',
                                         'hubspot', 'zoho', 'freshworks', 'cloudflare', 'digitalocean'],
            'Education': ['school', 'college', 'university', 'coaching', 'course', 'udemy', 'byju',
                          'unacademy', 'tuition', 'coursera', 'skillshare', 'vedantu'],
            'Rent & Housing': ['rent', 'lease', 'housing', 'society', 'maintenance', 'flat', 'apartment',
                                'pg payment', 'hostel', 'nobroker'],
            'Salary & Income': ['salary', 'payroll', 'income', 'wages', 'stipend', 'bonus',
                                 'incentive', 'commission'],
            'Transfers': ['transfer', 'imps', 'rtgs', 'sent to', 'received from', 'neft cr', 'neft dr'],
            'ATM & Cash': ['atm', 'cash withdrawal', 'cash deposit', 'chq dep', 'cheque deposit',
                           'cash dep', 'cdm'],
            'Insurance': ['insurance', 'lic', 'premium', 'policy', 'bajaj allianz', 'hdfc ergo',
                           'icici lombard', 'star health', 'care health'],
            'Investment & Savings': ['mutual fund', 'zerodha', 'groww', 'upstox', 'stocks', 'sip',
                                     'ppf', 'nps', 'investment', 'smallcase'],
            'Taxes & Government': ['tax', 'gst', 'income tax', 'govt', 'challan', 'tds', 'mca',
                                    'epfo', 'profession tax', 'pt payment', 'esi', 'icegate',
                                    'customs duty', 'advance tax', 'itns 280', 'itns 281'],
            'EMI & Loan Repayment': ['emi', 'loan emi', 'home loan', 'car loan', 'personal loan',
                                      'nach dr', 'enach', 'ecs debit', 'loan repayment', 'term loan'],
            'Professional & Legal Fees': ['consulting', 'legal', 'audit', 'accounting', 'lawyer',
                                           'chartered accountant', 'ca firm', 'advocate'],
            'Marketing & Advertising': ['google ads', 'facebook ads', 'advertising', 'marketing',
                                         'meta ads', 'instagram ads', 'linkedin ads', 'indiamart'],
        }

        for rule_cat, keywords in builtin_rules.items():
            matched_cat = next(
                (c for c in categories if rule_cat.lower() in c.lower() or c.lower() in rule_cat.lower()),
                None
            )
            if not matched_cat:
                continue
            for keyword in keywords:
                if keyword in clean_desc:
                    score = min(0.85 + (len(keyword) / max(len(clean_desc), 1)) * 0.1, 0.93)
                    if score > best_score:
                        best_score = score
                        best_match = matched_cat

        return best_match, best_score

    def ai_classify(self, description: str, categories: List[str],
                    amount: Optional[float] = None, context: str = '') -> Tuple[Optional[str], float]:
        if not self.client or not description:
            return None, 0.0
        try:
            category_str = '\n'.join(f'- {c}' for c in categories[:60])
            amount_str   = f'\nTransaction Amount: ₹{amount:,.2f}' if amount else ''
            context_str  = f'\nFile context: {context}' if context else ''

            system_prompt = (
                'You are a senior Chartered Accountant (CA) with 20+ years of experience in Indian '
                'bookkeeping, GST compliance, TDS/TCS, and financial reporting under the Companies Act 2013 '
                '(Schedule III). You ALWAYS return valid JSON only — no markdown, no explanation outside the JSON.'
            )
            prompt = (
                f'Classify the following Indian bank transaction narration into exactly one category from the list below.\n\n'
                f'=== TRANSACTION ===\n'
                f'Narration: "{description}"{amount_str}{context_str}\n\n'
                f'=== AVAILABLE CATEGORIES ===\n{category_str}\n\n'
                f'=== RULES ===\n'
                f'1. Parse vendor names, UPI handles, and payment codes carefully.\n'
                f'2. Map TDS/GST/PF/ESI to the correct statutory head.\n'
                f'3. Choose the MOST SPECIFIC category. Avoid "Other" unless truly unclassifiable.\n'
                f'4. Confidence: 0.92+ very certain | 0.75-0.91 fairly sure | 0.55-0.74 educated guess.\n\n'
                f'Respond ONLY with this JSON:\n'
                f'{{"category": "exact name from list", "confidence": 0.85, "reasoning": "one concise sentence"}}'
            )

            chat = self.client.chat.completions.create(
                model=self.openrouter_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': prompt}
                ],
                temperature=0.1,
                max_tokens=300,
            )
            response_text = chat.choices[0].message.content.strip()
            response_text = re.sub(r'^```json\s*|\s*```$', '', response_text).strip()
            response_text = re.sub(r'^```\s*|\s*```$', '', response_text).strip()

            result     = json.loads(response_text)
            category   = result.get('category', 'Other')
            confidence = float(result.get('confidence', 0.5))

            if category not in categories:
                match = next((c for c in categories if category.lower() in c.lower()), None)
                if not match:
                    match = next((c for c in categories if any(
                        w in c.lower() for w in category.lower().split() if len(w) > 3
                    )), None)
                category = match if match else 'Other'
                if not match:
                    confidence = max(confidence - 0.15, 0.3)

            return category, confidence

        except json.JSONDecodeError as e:
            print(f"❌ [AI] JSON parse error: {e}")
            return None, 0.0
        except Exception as e:
            print(f"❌ [AI] ai_classify failed: {type(e).__name__}: {e}")
            return None, 0.0

    def classify_single(self, description: str, categories: List[str],
                        amount=None, use_ai: bool = True, context: str = '') -> Tuple[str, float, str]:
        category, confidence = self.keyword_match(description, categories)
        method = 'keyword'

        if use_ai and self.client and (category is None or confidence < 0.80):
            ai_cat, ai_conf = self.ai_classify(description, categories, amount, context)
            if ai_cat and ai_conf > (confidence or 0):
                category, confidence, method = ai_cat, ai_conf, 'ai'

        if category is None or confidence < 0.35:
            raw_lower = str(description).lower()
            if any(x in raw_lower for x in ['cash deposit', 'cdm deposit', 'chq dep', 'cheque dep', 'cash dep']):
                category, confidence, method = 'ATM & Cash', 0.72, 'keyword'
            elif any(x in raw_lower for x in ['neft cr', 'imps cr', 'rtgs cr', 'neft dr', 'imps dr', 'rtgs dr']):
                category, confidence, method = 'Transfers', 0.65, 'keyword'
            else:
                category   = category or 'Other'
                confidence = confidence or 0.40
                method     = method or 'none'

        return category, confidence, method

    def find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        cols_lower = {str(c).lower().strip(): c for c in df.columns}
        for name in possible_names:
            if name.lower() in cols_lower:
                return cols_lower[name.lower()]
        for name in possible_names:
            for col_lower, col in cols_lower.items():
                if name.lower() in col_lower:
                    return col
        return None

    def classify_transactions(self, df: pd.DataFrame, categories: List[str] = None,
                               use_ai: bool = True, confidence_threshold: float = 0.6,
                               progress_queue=None):
        file_type = self.detect_file_type(df)

        if not categories:
            if self.categories_loaded:
                categories = [str(cat.get('Category', '')) for cat in self.categories if cat.get('Category')]
            else:
                categories = self.get_auto_categories(file_type)

        desc_col = self.find_column(df, [
            'description', 'narration', 'desc', 'particulars', 'details',
            'merchant', 'payee', 'remarks', 'transaction details', 'narrations'
        ])
        amount_col = self.find_column(df, ['amount', 'debit', 'withdrawal', 'credit', 'value'])
        date_col   = self.find_column(df, ['date', 'transaction date', 'txn date', 'value date'])

        if desc_col is None:
            for col in df.columns:
                if df[col].dtype == object or df[col].astype(str).str.len().mean() > 5:
                    desc_col = col
                    break
            if desc_col is None:
                raise ValueError("Could not find a description/narration column")

        result_categories, confidences, methods, statuses = [], [], [], []
        codes, gst_flags, tds_sections, boe_flags = [], [], [], []
        context = f'File type: {file_type}'

        for idx, row in df.iterrows():
            description = str(row[desc_col]) if desc_col else ''
            amount = None
            if amount_col:
                try:
                    raw = str(row[amount_col]).replace(',', '').replace('₹', '').replace('$', '').strip()
                    val = float(raw)
                    amount = None if math.isnan(val) else val
                except Exception:
                    pass

            cat, conf, meth = self.classify_single(description, categories, amount, use_ai, context)

            # Look up meta for output columns
            meta = self.category_meta.get(cat, {})
            codes.append(meta.get('code', ''))
            gst_flags.append(meta.get('gst_flag', ''))
            tds_sections.append(meta.get('tds_section', ''))
            boe_flags.append(meta.get('boe_flag', ''))

            result_categories.append(cat)
            confidences.append(round(conf, 3))
            methods.append(meth)
            statuses.append('Approved' if conf >= confidence_threshold else 'Needs Review')

            processed = len(result_categories)
            if progress_queue is not None:
                progress_queue.put({
                    'type': 'progress',
                    'processed': processed,
                    'total': len(df),
                    'category': cat,
                    'method': meth,
                    'confidence': round(conf, 3),
                })
            elif processed % 25 == 0:
                print(f'  Processed {processed}/{len(df)}…')

        results = df.copy()
        results['Category']    = result_categories
        results['Code']        = codes
        results['Confidence']  = confidences
        results['Method']      = methods
        results['GST_Flag']    = gst_flags
        results['TDS_Section'] = tds_sections
        results['BOE_Flag']    = boe_flags
        results['Status']      = statuses

        if amount_col and 'Amount' not in results.columns:
            results['Amount'] = pd.to_numeric(
                results[amount_col].astype(str)
                    .str.replace(',', '', regex=False)
                    .str.replace('₹', '', regex=False)
                    .str.replace('$', '', regex=False)
                    .str.strip(),
                errors='coerce'
            )

        print('✅ Classification complete!')
        return results, file_type, categories

    def learn_mapping(self, description: str, category: str):
        vendor = self.extract_vendor(description)
        if vendor:
            self.vendor_memory[vendor] = category
            self.save_vendor_memory()