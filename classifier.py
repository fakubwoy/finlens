import pandas as pd
import re
import json
import os
import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class ExpenseClassifier:
    def __init__(self):
        self.categories = []
        self.category_keywords = {}
        self.vendor_memory = {}
        self.categories_loaded = False
        self.memory_file = 'data/vendor_memory.json'
        self.client = None
        self.ai_provider = None

        self.load_vendor_memory()
        self._init_ai()

    def _init_ai(self):
        """Initialise OpenRouter as the sole AI backend."""
        or_key = os.environ.get('OPENROUTER_API_KEY')
        if not or_key:
            print("⚠️  [AI] OPENROUTER_API_KEY not set — keyword-only mode active")
            print("   → Add OPENROUTER_API_KEY=<key> to your .env file to enable AI classification")
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
            # Default model — can be overridden via env var
            self.openrouter_model = os.environ.get(
                'OPENROUTER_MODEL',
                'meta-llama/llama-3.3-70b-instruct'   # fast & free-tier friendly
            )
            self.ai_provider = 'openrouter'
            print(f"✅ [AI] OpenRouter initialised | model: {self.openrouter_model}")
        except ImportError:
            print("⚠️  [AI] openai package not installed — run: pip install openai")
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

    def load_categories(self, df: pd.DataFrame):
        """Load from a simple Category/Keywords DataFrame."""
        self.categories = df.to_dict('records')
        for cat in self.categories:
            category_name = str(cat.get('Category', '') or '')
            keywords_raw = cat.get('Keywords', '') or ''
            keywords = str(keywords_raw).strip()
            if keywords and keywords.lower() not in ('nan', 'none', ''):
                self.category_keywords[category_name] = [k.strip().lower() for k in keywords.split(',')]
            else:
                self.category_keywords[category_name] = []
        self.categories_loaded = True
        print(f"✅ Loaded {len(self.categories)} categories")

    def load_categories_from_master(self, df: pd.DataFrame, category_names: List[str]):
        """
        Load from a chart-of-accounts master file.
        Tries to extract LLM keywords from 'LLM Keywords' / 'repeatingKeywords' columns.
        """
        self.categories = [{'Category': c} for c in category_names]
        self.category_keywords = {c: [] for c in category_names}

        cols_lower = {str(c).lower().strip(): c for c in df.columns}

        # Find keyword columns
        kw_cols = []
        for possible in ['llm keywords', 'llmkeywords', 'repeatingkeywords', 'repeating keywords', 'keywords']:
            if possible in cols_lower:
                kw_cols.append(cols_lower[possible])

        # Find account name column (to map keywords to category)
        name_col = None
        for p in ['account name', 'accountname', 'account_name', 'classification', 'name']:
            if p in cols_lower:
                name_col = cols_lower[p]
                break

        if name_col and kw_cols:
            for _, row in df.iterrows():
                cat_name = str(row.get(name_col, '') or '').strip()
                if not cat_name or cat_name.lower() in ('nan', ''):
                    continue

                # Find the closest matching category
                matched_cat = next(
                    (c for c in category_names if cat_name.lower() == c.lower()),
                    None
                )
                if not matched_cat:
                    # Fuzzy: cat_name contained in category
                    matched_cat = next(
                        (c for c in category_names if cat_name.lower() in c.lower() or c.lower() in cat_name.lower()),
                        None
                    )
                if not matched_cat:
                    continue

                for kw_col in kw_cols:
                    raw = str(row.get(kw_col, '') or '').strip()
                    if raw and raw.lower() not in ('nan', 'none', ''):
                        kws = [k.strip().lower() for k in re.split(r'[,;]', raw) if k.strip()]
                        self.category_keywords[matched_cat] = list(
                            set(self.category_keywords.get(matched_cat, []) + kws)
                        )

        self.categories_loaded = True
        total_kws = sum(len(v) for v in self.category_keywords.values())
        print(f"✅ Loaded {len(category_names)} categories from master, {total_kws} keywords")

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
        """
        Return default categories aligned with Indian accounting conventions:
        - Personal: common household / lifestyle heads
        - Business: aligned with Schedule III (Companies Act 2013) / Tally ledger groups
        """
        if file_type == 'bank_statement':
            return [
                # Income
                'Salary & Wages', 'Business Income', 'Interest Income', 'Dividend Income',
                # Transfers & Cash
                'Bank Transfer (NEFT/IMPS/RTGS)', 'UPI Transfer', 'ATM Withdrawal', 'Cheque Deposit',
                # Day-to-day
                'Food & Dining', 'Groceries & Kirana', 'Shopping & Retail',
                'Transport & Commute', 'Fuel & Petrol',
                # Housing & Utilities
                'Rent & Housing', 'Electricity & Power', 'Water & Municipal',
                'Mobile & Internet Recharge', 'DTH & Cable TV',
                # Finance
                'EMI & Loan Repayment', 'Credit Card Payment', 'Insurance Premium',
                'Mutual Fund / SIP', 'Investment & Shares', 'Fixed Deposit',
                # Taxes
                'Income Tax (TDS/Advance)', 'GST Payment', 'Professional Tax',
                # Govt / Compliance
                'PF / ESI Contribution', 'Govt Fees & Challan',
                # Personal
                'Healthcare & Medical', 'Education & Coaching', 'Entertainment & OTT',
                'Travel & Hotel', 'Personal Care & Salon',
                # Misc
                'Other',
            ]
        elif file_type == 'business_transactions':
            # Aligned with Schedule III P&L heads and Tally ledger groups
            return [
                # Revenue
                'Sales / Revenue', 'Service Income', 'Other Operating Income',
                # Direct costs
                'Purchase of Stock-in-Trade', 'Raw Material Purchase', 'Freight Inward',
                # Employee costs
                'Salaries & Wages', 'Director Remuneration', 'ESI & PF Contribution',
                'Staff Welfare', 'Bonus & Ex-Gratia',
                # Administrative expenses
                'Rent & Lease', 'Office Supplies & Stationery', 'Printing & Stationery',
                'Postage & Courier', 'Telephone & Internet', 'Electricity Charges',
                'Repairs & Maintenance', 'Vehicle Running Expenses',
                # Finance costs
                'Bank Charges & Commission', 'Loan Interest', 'EMI & Loan Repayment',
                'Credit Card Payment',
                # Statutory & Compliance
                'GST Payment', 'TDS / TCS Deposit', 'Income Tax Payment',
                'Professional Tax', 'ROC / MCA Fees', 'PF / ESI Deposit',
                # Sales & Marketing
                'Advertising & Marketing', 'Business Promotion', 'Commission & Brokerage',
                # Professional services
                'Professional & Legal Fees', 'Audit & Accounting Fees', 'Consultancy Charges',
                # Technology
                'Software & SaaS Subscriptions', 'Cloud Infrastructure (AWS/Azure/GCP)',
                'IT & Hardware Purchase',
                # Capital / Investments
                'Capital Expenditure', 'Investment in Securities',
                # Travel
                'Travel & Conveyance', 'Hotel & Accommodation', 'Meals & Entertainment',
                # Insurance
                'Insurance Premium',
                # Transfers
                'Intercompany Transfer', 'Bank Transfer',
                # Misc
                'Miscellaneous Expense', 'Other',
            ]
        else:
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
        # Strip common Indian bank prefixes that carry no semantic meaning
        desc = re.sub(
            r'^(upi|neft|imps|rtgs|pos|atm|trf|ach|wire|nach|si|enach|'
            r'me dc si|billdesk|razorpay\*|payu\*|ccavenue\*|cashfree\*|'
            r'paytm\*|phonepe\*|gpay\*|googlepay\*|mobikwik\*|freecharge\*|'
            r'iob|sbi|hdfc|icici|axis|kotak|yes bank|bob|pnb|'
            r'cdm deposit|cash dep|chq dep|cheque dep)\s*[-:/\*]?\s*',
            '', desc
        )
        desc = re.sub(r'\b[a-z0-9]{12,}\b', '', desc)   # remove long reference codes
        desc = re.sub(r'\b\d{6,}\b', '', desc)           # remove long numeric refs
        desc = re.sub(r'[^a-z0-9\s&]', ' ', desc)
        desc = re.sub(r'\s+', ' ', desc).strip()
        return desc

    def extract_vendor(self, description: str) -> str:
        clean = self.preprocess_description(description)
        words = clean.split()
        return ' '.join(words[:min(3, len(words))]) if words else clean

    def keyword_match(self, description: str, categories: List[str]) -> Tuple[Optional[str], float]:
        clean_desc = self.preprocess_description(description)

        vendor = self.extract_vendor(description)
        if vendor and vendor in self.vendor_memory:
            return self.vendor_memory[vendor], 0.95

        best_match = None
        best_score = 0.0

        # Check loaded category keywords first (from master file)
        for category, keywords in self.category_keywords.items():
            if category not in categories:
                continue
            for keyword in keywords:
                if keyword and keyword in clean_desc:
                    score = 0.85 + (len(keyword) / max(len(clean_desc), 1)) * 0.1
                    score = min(score, 0.94)
                    if score > best_score:
                        best_score = score
                        best_match = category

        if best_match:
            return best_match, best_score

        # Built-in fallback keyword rules
        builtin_rules = {
            'Food & Dining': ['swiggy', 'zomato', 'restaurant', 'cafe', 'food', 'pizza', 'burger',
                              'bakery', 'starbucks', 'mcdonalds', 'kfc', 'dominos', 'dining', 'canteen',
                              'dunzo', 'zepto food', 'blinkit food', 'haldiram', 'subway', 'barbeque'],
            'Shopping & Retail': ['amazon', 'flipkart', 'myntra', 'nykaa', 'ajio', 'shop', 'store',
                                  'mart', 'retail', 'mall', 'bazaar', 'bigbasket', 'blinkit', 'zepto',
                                  'vijay sales', 'vijaysales', 'croma', 'reliance digital', 'chroma',
                                  'meesho', 'snapdeal', 'tatacliq', 'd mart', 'dmart', 'payuamazon',
                                  'payuvijay', 'shopify', 'jiomart'],
            'Transport & Travel': ['uber', 'ola', 'rapido', 'redbus', 'irctc', 'airline', 'flight',
                                   'train', 'bus', 'taxi', 'cab', 'metro', 'petrol', 'fuel', 'makemytrip',
                                   'easytrip', 'easymytrip', 'indigo', 'spicejet', 'vistara', 'airindia',
                                   'air india', 'goair', 'akasa', 'yatra', 'cleartrip', 'goibibo',
                                   'payviarazorpay', 'indian railways', 'railways cate', 'namma metro',
                                   'bmtc', 'tsrtc', 'ksrtc', 'msrtc', 'hp petrol', 'iocl', 'bpcl'],
            'Utilities & Bills': ['electricity', 'water', 'gas', 'broadband', 'airtel', 'jio', 'bsnl',
                                  'vi ', 'vodafone', 'internet', 'mobile', 'recharge', 'bill', 'tata power',
                                  'bescom', 'msedcl', 'tneb', 'cesc', 'adani electricity', 'mahanagar gas',
                                  'igl', 'mgl', 'act fibernet', 'hathway', 'tikona'],
            'Healthcare & Medical': ['hospital', 'clinic', 'pharmacy', 'medical', 'doctor', 'health',
                                     'medicine', 'lab', 'diagnostic', 'apollo', 'fortis', 'dental',
                                     'medplus', 'netmeds', 'pharmeasy', '1mg', 'tata 1mg', 'practo',
                                     'thyrocare', 'dr lal', 'srl diagnostic', 'eye hospital', 'superspeciality',
                                     'orthopedic', 'pathology', 'radiology'],
            'Entertainment': ['netflix', 'amazon prime', 'hotstar', 'spotify', 'youtube', 'gaming',
                               'cinema', 'movie', 'pvr', 'inox', 'bookmyshow', 'disney', 'zee5',
                               'sonyliv', 'jiocinema', 'voot', 'mxplayer', 'alt balaji', 'lionsgate'],
            'Software & Subscriptions': ['aws', 'azure', 'google cloud', 'github', 'slack', 'zoom',
                                         'microsoft', 'adobe', 'salesforce', 'saas', 'chatgpt', 'openai',
                                         'googleworksp', 'google workspace', 'gsuite', 'g suite',
                                         'dropbox', 'notion', 'figma', 'canva', 'hubspot', 'zoho',
                                         'freshworks', 'razorpay subscr', 'stripe', 'twilio', 'sendgrid',
                                         'datadog', 'new relic', 'cloudflare', 'digitalocean', 'heroku',
                                         'vercel', 'netlify', 'godaddy', 'hostinger', 'namecheap'],
            'Subscriptions': ['netflix subscr', 'spotify subscr', 'prime subscr', 'chatgpt subscr',
                              'openai subscr', 'adobe subscr', 'microsoft subscr', 'annual plan',
                              'monthly plan', 'auto renew', 'auto-renew', 'recurring'],
            'Education': ['school', 'college', 'university', 'coaching', 'course', 'udemy', 'byju',
                          'unacademy', 'tuition', 'fees', 'coursera', 'skillshare', 'linkedin learning',
                          'great learning', 'simplilearn', 'vedantu', 'toppr', 'khan academy'],
            'Rent & Housing': ['rent', 'lease', 'housing', 'society', 'maintenance', 'flat', 'apartment',
                                'pg payment', 'hostel', 'nobroker', 'magicbricks', 'stanza'],
            'Salary & Income': ['salary', 'payroll', 'income', 'wages', 'stipend', 'bonus',
                                 'incentive', 'commission', 'dr vijay incentive', 'consultant fee'],
            'Transfers': ['transfer', 'imps', 'rtgs', 'sent to', 'received from', 'self transfer',
                           'own account', 'neft cr', 'neft dr'],
            'ATM & Cash': ['atm', 'cash withdrawal', 'cash deposit', 'chq dep', 'cheque deposit',
                           'cash dep', 'cdm', 'white label atm'],
            'Insurance': ['insurance', 'lic', 'premium', 'policy', 'bajaj allianz', 'hdfc ergo',
                           'icici lombard', 'star health', 'care health', 'niva bupa', 'tata aig'],
            'Investment & Savings': ['mutual fund', 'zerodha', 'groww', 'upstox', 'stocks', 'sip',
                                     'ppf', 'nps', 'investment', 'smallcase', 'coin by zerodha',
                                     'motilal oswal', 'iifl securities', 'angel broking', 'icici direct',
                                     'hdfc securities', 'kotak securities'],
            'Taxes & Government': ['tax', 'gst', 'income tax', 'govt', 'challan', 'tds', 'mca',
                                    'epfo', 'pf deposit', 'profession tax', 'pt payment', 'esi',
                                    'icegate', 'customs duty', 'advance tax', 'self assessment tax',
                                    'itns 280', 'itns 281', 'tds challan', 'gst challan'],
            'PF / ESI Contribution': ['epfo', 'pf contribution', 'esi contribution', 'provident fund',
                                       'employee state insurance', 'pf deposit', 'epf'],
            'GST Payment': ['gst challan', 'gstin', 'cgst', 'sgst', 'igst', 'gst payment',
                            'goods and services tax', 'icegate gst'],
            'Income Tax (TDS/Advance)': ['tds deposit', 'tds payment', 'advance tax', 'self assessment',
                                          'itns 280', 'itns 281', 'income tax refund', 'it refund'],
            'Professional Tax': ['profession tax', 'professional tax', 'pt payment', 'pt challan'],
            # Business categories
            'Office Supplies': ['stationery', 'office depot', 'paper', 'printer', 'supplies', 'pen drive',
                                 'staples', 'acco', 'camlin', 'faber castell'],
            'Marketing & Advertising': ['google ads', 'facebook ads', 'advertising', 'marketing', 'campaign',
                                         'meta ads', 'instagram ads', 'linkedin ads', 'just dial', 'sulekha',
                                         'indiamart', 'tradeindia'],
            'Professional & Legal Fees': ['consulting', 'legal', 'audit', 'accounting', 'lawyer', 'advisory',
                                           'chartered accountant', 'ca firm', 'advocate', 'notary',
                                           'legal fees', 'retainer'],
            'Payroll & Salaries': ['salary', 'payroll', 'wages', 'staff', 'employee', 'esi contribution',
                                    'monthly salary', 'staff salary', 'worker wages'],
            'Equipment & Hardware': ['laptop', 'computer', 'server', 'hardware', 'equipment', 'machinery',
                                     'dell', 'hp laptop', 'lenovo', 'apple macbook', 'ipad', 'iphone purchase',
                                     'printer purchase', 'scanner', 'projector'],
            'EMI & Loan Repayment': ['emi', 'loan emi', 'home loan', 'car loan', 'personal loan',
                                      'nach dr', 'enach', 'ecs debit', 'loan repayment', 'term loan',
                                      'hdfc bank loan', 'icici loan', 'bajaj finance emi', 'capital float'],
            'Bank Transfer (NEFT/IMPS/RTGS)': ['neft', 'rtgs', 'imps', 'bank transfer', 'inter bank',
                                                 'fund transfer', 'self transfer', 'own account'],
        }

        for rule_cat, keywords in builtin_rules.items():
            # Match to actual category list (fuzzy)
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
            category_str = '\n'.join(f'- {c}' for c in categories[:60])  # cap at 60
            amount_str = f'\nTransaction Amount: ₹{amount:,.2f}' if amount else ''
            context_str = f'\nFile context: {context}' if context else ''

            system_prompt = (
                'You are a senior Chartered Accountant (CA) with 20+ years of experience in Indian '
                'bookkeeping, GST compliance, TDS/TCS, and financial reporting under the Companies Act 2013 '
                '(Schedule III). You specialise in decoding cryptic Indian bank narrations — UPI, NEFT, IMPS, '
                'RTGS, POS, standing instructions, payment gateways (Razorpay, PayU, CCAvenue, Cashfree), '
                'and reconciling them to appropriate Tally / ERP ledger heads. '
                'You ALWAYS return valid JSON only — no markdown, no explanation outside the JSON.'
            )

            prompt = (
                f'Classify the following Indian bank transaction narration into exactly one category from the list below.\n\n'
                f'=== TRANSACTION ===\n'
                f'Narration: "{description}"{amount_str}{context_str}\n\n'
                f'=== AVAILABLE CATEGORIES ===\n{category_str}\n\n'
                f'=== CLASSIFICATION RULES ===\n'
                f'1. Parse the narration carefully — vendor names, UPI handles, and payment codes are key.\n'
                f'2. Indian bank narration patterns:\n'
                f'   • "UPI/<vendor>/<upiid>@<bank>" or "UPI-<vendor>-<ref>" → classify by vendor\n'
                f'   • "NEFT CR/<ifsc>/<sender>" → credit from sender (salary, income, transfer)\n'
                f'   • "NEFT DR/<ifsc>/<receiver>" → payment to receiver\n'
                f'   • "IMPS/<ref>/<name>" → peer-to-peer or vendor transfer\n'
                f'   • "RTGS/<ref>/<name>" → large-value transfer (>₹2L), likely B2B or loan\n'
                f'   • "POS/<terminal>/<merchant>" → in-store card swipe at merchant\n'
                f'   • "SI/<mandate>" or "ME DC SI" → standing instruction (EMI, subscription, SIP)\n'
                f'   • "NACH DR/<company>" → auto-debit mandate (insurance premium, EMI, SIP)\n'
                f'   • "CDM DEPOSIT" / "CASH DEP" → cash deposit at CDM/branch\n'
                f'   • "CHQ DEP/<chq no>" → cheque deposit\n'
                f'   • "RAZORPAY*" / "PAYU*" / "CCAVENUE*" → online payment; decode merchant after *\n'
                f'   • "BILLDESK/<biller>" → utility bill payment (electricity, gas, water, telecom)\n'
                f'   • "ENACH DR/<company>" → recurring mandate (LIC, mutual fund, EMI)\n'
                f'3. Statutory / compliance heads (map carefully):\n'
                f'   • TDS/TCS challan → Income Tax (TDS/Advance)\n'
                f'   • GST challan (GSTIN ref, ICEGATE) → GST Payment\n'
                f'   • EPFO/PF/ESI → PF / ESI Contribution or Deposit\n'
                f'   • PT challan → Professional Tax\n'
                f'   • MCA/ROC → ROC / MCA Fees\n'
                f'4. Vendor shortcuts (well-known in India):\n'
                f'   IRCTC→Travel; Swiggy/Zomato→Food; Amazon/Flipkart→Shopping;\n'
                f'   AWS/Azure/GCP→Cloud; Uber/Ola/Rapido→Transport; Zerodha/Groww/Coin→Investment;\n'
                f'   LIC/StarHealth/Niva Bupa→Insurance; VIJAYSALES/Croma→Electronics;\n'
                f'   BESCOM/MSEDCL/TNEB→Electricity; JIO/Airtel/BSNL→Mobile & Internet;\n'
                f'   Mahanagar Gas/IGL/MGL→Piped Gas; HDFC/ICICI/SBI (bank name alone)→Bank Transfer.\n'
                f'5. Amount heuristics (use only when narration is ambiguous):\n'
                f'   • >₹1,00,000 with no clear vendor → likely Bank Transfer or EMI & Loan\n'
                f'   • ₹10,000–₹50,000 credit from individual name → Salary or Transfers\n'
                f'   • Round amounts (₹500, ₹1000) via UPI → likely Transfers or Bill Payment\n'
                f'6. Choose the MOST SPECIFIC category. Avoid "Other" unless truly unclassifiable.\n'
                f'7. Confidence guide: 0.92+ very certain | 0.75–0.91 fairly sure | 0.55–0.74 educated guess.\n\n'
                f'Respond ONLY with this JSON (no markdown, no extra text):\n'
                f'{{"category": "exact name from list", "confidence": 0.85, "reasoning": "one concise sentence"}}'
            )

            # OpenRouter — OpenAI-compatible endpoint
            chat = self.client.chat.completions.create(
                model=self.openrouter_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.1,
                max_tokens=300,
            )
            response_text = chat.choices[0].message.content.strip()

            response_text = re.sub(r'^```json\s*|\s*```$', '', response_text).strip()
            response_text = re.sub(r'^```\s*|\s*```$', '', response_text).strip()

            result = json.loads(response_text)
            category = result.get('category', 'Other')
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
            print(f"❌ [AI] JSON parse error: {e} | Response was: {response_text[:200]}")
            return None, 0.0
        except Exception as e:
            import traceback
            print(f"❌ [AI] ai_classify failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None, 0.0

    def classify_single(self, description: str, categories: List[str],
                        amount=None, use_ai: bool = True, context: str = '') -> Tuple[str, float, str]:
        category, confidence = self.keyword_match(description, categories)
        method = 'keyword'

        short_desc = description[:60] if description else ''

        if not use_ai:
            print(f"  [CLASSIFY] AI disabled for: {short_desc!r}")
        elif not self.client:
            print(f"  [CLASSIFY] No AI client — keyword only for: {short_desc!r}")
        elif category is not None and confidence >= 0.80:
            print(f"  [CLASSIFY] Keyword high-conf ({confidence:.2f}) '{category}' — skipping AI for: {short_desc!r}")
        else:
            kw_info = f"kw=({category},{confidence:.2f})" if category else "kw=(none)"
            print(f"  [CLASSIFY] Trying AI {kw_info} for: {short_desc!r}")
            ai_cat, ai_conf = self.ai_classify(description, categories, amount, context)
            print(f"  [CLASSIFY] AI returned: cat={ai_cat!r} conf={ai_conf:.2f}")
            if ai_cat and ai_conf > (confidence or 0):
                print(f"  [CLASSIFY] AI wins: {ai_cat!r} ({ai_conf:.2f}) > keyword ({confidence:.2f})")
                category, confidence, method = ai_cat, ai_conf, 'ai'
            else:
                print(f"  [CLASSIFY] Keyword wins or AI failed: keeping {category!r} ({confidence:.2f})")

        # Last-resort pattern fallbacks for common Indian bank narrations
        if category is None or confidence < 0.35:
            raw_lower = str(description).lower()
            if any(x in raw_lower for x in ['cash deposit', 'cdm deposit', 'chq dep', 'cheque dep', 'cash dep']):
                category, confidence, method = 'ATM & Cash', 0.72, 'keyword'
                print(f"  [CLASSIFY] Fallback → ATM & Cash for: {short_desc!r}")
            elif any(x in raw_lower for x in ['neft cr', 'imps cr', 'rtgs cr']):
                category, confidence, method = 'Transfers', 0.65, 'keyword'
                print(f"  [CLASSIFY] Fallback → Transfers (credit) for: {short_desc!r}")
            elif any(x in raw_lower for x in ['neft dr', 'imps dr', 'rtgs dr']):
                category, confidence, method = 'Transfers', 0.65, 'keyword'
                print(f"  [CLASSIFY] Fallback → Transfers (debit) for: {short_desc!r}")
            else:
                category = category or 'Other'
                confidence = confidence or 0.40
                method = method or 'none'
                print(f"  [CLASSIFY] Final fallback → Other for: {short_desc!r}")

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
                               use_ai: bool = True,
                               confidence_threshold: float = 0.6,
                               progress_queue=None):
        print(f"\n{'='*60}")
        print(f"[CLASSIFY] Starting classification")
        print(f"[CLASSIFY] use_ai={use_ai}")
        print(f"[CLASSIFY] AI provider: {self.ai_provider!r}")
        print(f"[CLASSIFY] AI client set: {self.client is not None}")
        print(f"[CLASSIFY] Categories loaded: {self.categories_loaded}")
        print(f"[CLASSIFY] Row count: {len(df)}")
        print(f"{'='*60}")
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
        amount_col = self.find_column(df, ['amount', 'debit', 'withdrawal', 'credit', 'value', 'dr amount', 'cr amount'])
        date_col = self.find_column(df, ['date', 'transaction date', 'txn date', 'value date', 'posting date'])

        if desc_col is None:
            # Fall back to first object column
            for col in df.columns:
                if df[col].dtype == object or df[col].astype(str).str.len().mean() > 5:
                    desc_col = col
                    break
            if desc_col is None:
                raise ValueError("Could not find a description/narration column in your file")

        result_categories, confidences, methods, statuses = [], [], [], []
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
        results['Category'] = result_categories
        results['Confidence'] = confidences
        results['Method'] = methods
        results['Status'] = statuses

        # Add a clean Amount column if possible
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