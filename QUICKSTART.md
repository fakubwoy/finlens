# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1️⃣ Run the Application

**On Linux/Mac:**
```bash
./run.sh
```

**On Windows:**
```
run.bat
```

### 2️⃣ Open Your Browser

Navigate to: **http://localhost:5000**

### 3️⃣ Upload & Classify

1. **Upload Categories**: Click the left box and upload `sample_categories.csv`
2. **Upload Transactions**: Click the right box and upload `sample_transactions.csv`
3. **Click "Classify Transactions"** button
4. **View Results** and download the classified file

## 🎯 That's It!

Your transactions are now automatically categorized.

## ⚙️ Optional: Enable AI Classification

For better accuracy with unknown vendors using open-source LLMs:

1. Get a free API key from https://console.groq.com
2. Set the environment variable:
   ```bash
   export GROQ_API_KEY='your-key-here'
   ```
3. Restart the application

**Supported Models**: Llama 3.3 70B, Mixtral 8x7B, and more!

## 📚 Need More Help?

See the full README.md for detailed documentation.

## 🐛 Having Issues?

- Make sure Python 3.8+ is installed
- Check that port 5000 is available
- Try running with `python3 app.py` directly
- Review the troubleshooting section in README.md