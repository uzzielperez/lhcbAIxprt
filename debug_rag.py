#!/usr/bin/env python3
"""
Debug script to check RAG system functionality
"""

import pickle
import os
from pathlib import Path

def check_documents():
    """Check what documents are loaded in the system"""
    print("🔍 RAG System Debug")
    print("=" * 50)
    
    # Check if data file exists
    data_file = "shifter_docs_simple.pkl"
    if not os.path.exists(data_file):
        print(f"❌ No data file found: {data_file}")
        return
    
    print(f"✅ Data file found: {data_file}")
    
    try:
        # Load the data
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        documents = data.get('documents', [])
        chunks = data.get('chunks', [])
        chunk_metadata = data.get('chunk_metadata', [])
        
        print(f"\n📊 Statistics:")
        print(f"Documents: {len(documents)}")
        print(f"Text chunks: {len(chunks)}")
        print(f"Metadata entries: {len(chunk_metadata)}")
        
        if documents:
            print(f"\n📚 Loaded Documents:")
            for i, doc in enumerate(documents):
                print(f"{i+1}. {doc['filename']} ({doc['file_type']})")
                print(f"   Words: {doc['word_count']}, Size: {doc['size']} chars")
                print(f"   Uploaded: {doc['upload_time'][:19]}")
                print()
        
        if chunks:
            print(f"\n📄 Sample chunks (first 3):")
            for i, chunk in enumerate(chunks[:3]):
                print(f"Chunk {i+1}: {chunk[:100]}...")
                print()
        
        # Test search functionality
        if chunks:
            test_search(chunks, chunk_metadata, documents)
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")

def test_search(chunks, chunk_metadata, documents):
    """Test the search functionality"""
    print("🔍 Testing Search Functionality")
    print("-" * 30)
    
    # Test queries
    test_queries = [
        "system",
        "emergency",
        "restart",
        "alarm",
        "procedure"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        
        # Simple text search
        results = []
        query_words = set(query.lower().split())
        
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            intersection = query_words.intersection(chunk_words)
            
            if intersection:
                similarity = len(intersection) / len(query_words.union(chunk_words))
                results.append({
                    'chunk_id': i,
                    'similarity': similarity,
                    'content': chunk,
                    'metadata': chunk_metadata[i] if i < len(chunk_metadata) else {}
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        if results:
            print(f"  ✅ Found {len(results)} matches")
            print(f"  Top match (score: {results[0]['similarity']:.3f}):")
            print(f"    {results[0]['content'][:150]}...")
        else:
            print(f"  ❌ No matches found")

def test_groq_connection():
    """Test Groq API connection"""
    print("\n🤖 Testing Groq API Connection")
    print("-" * 30)
    
    try:
        from groq import Groq
        
        # Check if secrets file exists
        secrets_file = ".streamlit/secrets.toml"
        if not os.path.exists(secrets_file):
            print("❌ Secrets file not found")
            return
        
        # Try to read the API key
        with open(secrets_file, 'r') as f:
            content = f.read()
            if 'groq_api_key' in content and 'your_groq_api_key_here' not in content:
                print("✅ Groq API key configured")
                
                # Test basic API call
                try:
                    import re
                    api_key_match = re.search(r'groq_api_key\s*=\s*["\']([^"\']+)["\']', content)
                    if api_key_match:
                        api_key = api_key_match.group(1)
                        client = Groq(api_key=api_key)
                        
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": "Hello, this is a test."}],
                            max_tokens=50
                        )
                        print("✅ Groq API connection successful")
                        print(f"   Response: {response.choices[0].message.content[:100]}...")
                    else:
                        print("❌ Could not extract API key from secrets file")
                        
                except Exception as e:
                    print(f"❌ Groq API test failed: {e}")
            else:
                print("❌ Groq API key not properly configured")
                
    except ImportError:
        print("❌ Groq library not installed")
    except Exception as e:
        print(f"❌ Error testing Groq: {e}")

if __name__ == "__main__":
    check_documents()
    test_groq_connection()
    
    print("\n" + "=" * 50)
    print("🎯 Troubleshooting Tips:")
    print("1. Make sure documents were uploaded and processed successfully")
    print("2. Check that your query contains words that appear in the documents")
    print("3. Verify Groq API key is working")
    print("4. Try simple queries first (e.g., common words from your documents)")
    print("\n💡 If everything looks good, the issue might be in the Streamlit interface.")
    print("   Try refreshing the browser or restarting the app.") 