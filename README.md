# Automated Expense Classification System

AI-powered bank transaction categorization system that automatically classifies expenses using keyword matching and Claude AI.

## 🎯 Features

- **Automated Classification**: Classify bank transactions into predefined expense categories
- **Dual Classification Modes**: 
  - Keyword matching for known vendors
  - AI-powered classification using Claude for unknown transactions
- **Confidence Scoring**: Each classification includes a confidence score (0-1)
- **Smart Learning**: System learns from manual corrections to improve future classifications
- **Vendor Memory**: Automatically remembers vendor-to-category mappings
- **Professional Dashboard**: Beautiful web interface for uploading files and viewing results
- **Bulk Processing**: Handle thousands of transactions in one go
- **Export Results**: Download classified transactions as CSV

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Groq API key (optional, for AI classification using open-source LLMs)

## 🚀 Quick Start

### 1. Installation

Clone or download this project, then run:

```bash
./run.sh
```

The script will:
- Check for Python installation
- Create a virtual environment
- Install all dependencies
- Start the Flask server

### 2. Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

### 3. Upload Files

**Step 1: Upload Categories**
- Click on the "Upload Categories" area
- Upload a CSV/Excel file with your expense categories
- File should have at minimum a "Category" column
- Optional: Add "Code" and "Keywords" columns for better matching

Example format:
```csv
Code,Category,Keywords
5001,Office Rent,rent,lease,office
5002,Travel Expense,uber,ola,taxi,flight
5003,Software,google,aws,azure,cloud
```

**Step 2: Upload Transactions**
- Click on the "Upload Transactions" area
- Upload your bank statement (CSV/Excel)
- File should have columns for Date, Description, and Amount

Example format:
```csv
Date,Description,Amount
02-Jan-2024,UPI SWIGGY PAYMENT,550
03-Jan-2024,GOOGLE CLOUD INDIA,3500
```

**Step 3: Classify**
- Configure settings (AI classification, confidence threshold)
- Click "Classify Transactions"
- View results and download classified file

## 📁 Project Structure

```
expense-classifier/
├── app.py                      # Flask application
├── classifier.py               # Classification engine
├── requirements.txt            # Python dependencies
├── run.sh                      # Setup and run script
├── templates/
│   └── index.html             # Web interface
├── uploads/                    # Uploaded files storage
├── outputs/                    # Classification results
├── data/                       # Vendor memory and temp files
├── sample_categories.csv       # Example categories
└── sample_transactions.csv     # Example transactions
```

## 🔧 Configuration

### Environment Variables

**GROQ_API_KEY** (Optional - for AI classification)
```bash
export GROQ_API_KEY='your-key-here'
```

Get your free API key at: https://console.groq.com

Without this key, the system will use keyword matching only (still very effective!).

### Classification Settings

- **Use AI Classification**: Enable/disable Claude AI classification
- **Confidence Threshold**: Transactions below this threshold are flagged for review (default: 0.60)

## 📊 How It Works

### 1. Preprocessing
- Cleans transaction descriptions (removes IDs, special characters)
- Normalizes text (lowercase, removes noise)
- Extracts vendor names

### 2. Classification Process

**Step 1: Check Vendor Memory**
- If vendor has been classified before, reuse the category

**Step 2: Keyword Matching**
- Match description against predefined keywords
- Confidence: 0.85-0.95 for keyword matches

**Step 3: AI Classification** (if enabled and no keyword match)
- Use Groq AI (open-source LLMs like Llama 3.3) to infer the category
- Handles unknown/ambiguous transactions
- Confidence: Based on AI certainty

**Step 4: Assign Confidence**
- High confidence (≥0.7): Auto-approved
- Low confidence (<0.6): Flagged for review

### 3. Learning System
- Manual corrections are stored in vendor memory
- Future transactions from the same vendor use learned mapping
- Improves accuracy over time

## 📈 Output

The system generates a CSV file with:
- Original transaction data
- Assigned category
- Confidence score (0-1)
- Classification method (keyword/ai)
- Status (Approved/Needs Review)

Example:
```csv
Date,Description,Amount,Category,Confidence,Status
02-Jan,SWIGGY PAYMENT,550,Staff Welfare,0.94,Approved
03-Jan,GOOGLE CLOUD,3500,Software,0.96,Approved
05-Jan,XYZ CONSULTING,5000,Professional Fees,0.42,Needs Review
```

## 🎨 Sample Files

Two sample files are included for testing:

1. **sample_categories.csv**: Example expense categories with keywords
2. **sample_transactions.csv**: Example bank transactions

You can use these to test the system before uploading your own data.

## 🔒 Security Notes

- All files are processed locally
- No data is sent externally (except to Groq API if AI classification is enabled)
- Vendor memory is stored locally in `data/vendor_memory.json`
- Upload files are stored in the `uploads/` directory

## 🐛 Troubleshooting

### Port Already in Use
If port 5000 is busy, edit `app.py` and change the port:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### AI Classification Not Working
- Ensure GROQ_API_KEY is set
- Check API key is valid (get one free at https://console.groq.com)
- Verify network connectivity
- Alternative: Disable AI classification and use keyword-only mode

### File Upload Errors
- Check file format (CSV or Excel)
- Ensure required columns exist
- File size must be under 16MB
- Check file permissions

### Dependencies Installation Issues
If you encounter issues with pip, try:
```bash
pip install -r requirements.txt --break-system-packages
```

## 📝 API Endpoints

The system provides REST API endpoints:

- `POST /api/upload-categories` - Upload expense categories
- `POST /api/upload-transactions` - Upload bank transactions
- `POST /api/classify` - Classify all transactions
- `POST /api/update-classification` - Update single classification
- `GET /api/download-results` - Download classified results
- `GET /api/stats` - Get system statistics

## 🛠️ Manual Setup (Alternative to run.sh)

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key (optional)
export GROQ_API_KEY='your-api-key'

# Run application
python3 app.py
```

## 📄 License

This project is provided as-is for expense classification automation.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review sample files for format reference
3. Ensure all dependencies are installed
4. Verify Python version is 3.8+

## 🚀 Future Enhancements

Potential improvements:
- Multi-user support with authentication
- Database integration for persistence
- Advanced reporting and analytics
- Batch processing for multiple statements
- Custom rule creation interface
- Integration with accounting software
- Mobile responsive design improvements
- Real-time classification streaming

---

**Built with**: Python, Flask, Pandas, Groq AI (open-source LLMs: Llama 3.3, Mixtral)

**Version**: 1.0.0