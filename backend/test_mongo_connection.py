#!/usr/bin/env python3
"""
Test MongoDB Atlas connection to diagnose network access issues
"""
from pymongo import MongoClient
import sys

def test_mongodb_connection():
    print("🔍 Testing MongoDB Atlas Connection...")
    print(f"Your IP: 128.210.106.65")
    print("=" * 50)
    
    # Connection string from your notebook
    mongo_uri = "mongodb+srv://pranaynandkeolyar:nfl@cluster0.4nbxj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    
    try:
        print("Attempting connection...")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        
        # Test the connection
        client.admin.command('ping')
        print("✅ SUCCESS: Connected to MongoDB Atlas!")
        
        # List databases to confirm access
        dbs = client.list_database_names()
        print(f"📊 Available databases: {dbs}")
        
        # Test specific database access
        if 'ED' in dbs:
            ed_db = client['ED']
            collections = ed_db.list_collection_names()
            print(f"🏈 ED database collections: {len(collections)} found")
            if collections:
                print(f"   Sample collections: {collections[:3]}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ CONNECTION FAILED: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if your IP (128.210.106.65) is whitelisted in MongoDB Atlas")
        print("2. Go to Network Access in MongoDB Atlas dashboard")
        print("3. Add your IP address or use 0.0.0.0/0 for all IPs")
        print("4. Wait 2-3 minutes for changes to propagate")
        return False

if __name__ == "__main__":
    success = test_mongodb_connection()
    sys.exit(0 if success else 1)






