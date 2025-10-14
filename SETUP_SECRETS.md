# API Keys Setup Guide

## 🔐 Setting Up API Keys

To use the Shifter Assistant with vision capabilities, you need to configure your API keys in the secrets file.

### 1. Edit the Secrets File

Open `.streamlit/secrets.toml` and replace the placeholder values:

```toml
# Groq API Key (for text models)
groq_api_key = "your_actual_groq_api_key_here"

# Hugging Face API Key (for vision models)  
huggingface_api_key = "your_actual_huggingface_api_key_here"
```

### 2. Get Your API Keys

#### Groq API Key (for text models):
1. Go to https://console.groq.com/
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy the key

#### Hugging Face API Key (for vision models):
1. Go to https://huggingface.co/settings/tokens
2. Sign up or log in
3. Create a new token (free)
4. Copy the token

### 3. Security Notes

- **Never commit secrets to git** - The `.streamlit/secrets.toml` file should be in your `.gitignore`
- **Keep your API keys secure** - Don't share them publicly
- **Free tiers available** - Both services offer free usage limits

### 4. Test Your Setup

1. Start the app: `streamlit run shifter_rag_app_vision.py`
2. Check the sidebar for "✅ API Key Configured" messages
3. Use the "🔍 Test API Connection" buttons to verify

### 5. Troubleshooting

- **"API Key Required"** - Make sure you've added the keys to `secrets.toml`
- **"Connection failed"** - Check your internet connection and API key validity
- **"Model not found"** - Some models may be temporarily unavailable

## 🎯 Ready to Use!

Once configured, you can:
- Upload documents for text-based queries
- Upload images for AI-powered visual analysis
- Get context-aware responses based on your documentation
