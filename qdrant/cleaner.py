#!/usr/bin/env python3
"""
VectorDB Cleaner - Clean all documents from Qdrant collections
Provides options to:
1. Delete all points from specific collections
2. Delete entire collections
3. Recreate collections from scratch
4. Backup collections before cleaning
"""

import sys
from pathlib import Path
import json
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from qdrant.client import get_qdrant_client, setup_collections_dynamic, INTERNAL_COLLECTION, RFP_QA_COLLECTION
import settings

def print_separator(title: str, char: str = "=", length: int = 80):
    """Print a formatted separator with title"""
    print(f"\n{char * length}")
    print(f"{title:^{length}}")
    print(f"{char * length}")

def get_collection_info(client, collection_name: str) -> Dict[str, Any]:
    """Get information about a collection"""
    try:
        collection_info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points_count": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance": collection_info.config.params.vectors.distance,
            "exists": True
        }
    except Exception as e:
        return {
            "name": collection_name,
            "error": str(e),
            "exists": False
        }

def backup_collection(client, collection_name: str, backup_dir: Path) -> bool:
    """Backup a collection by exporting all points"""
    try:
        # Create backup directory
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all points
        points_result = client.scroll(
            collection_name=collection_name,
            limit=10000,  # Large limit to get all points
            with_payload=True,
            with_vectors=False  # Skip vectors to save space
        )
        
        points = points_result[0]
        if not points:
            print(f"⚠️  Collection '{collection_name}' is empty, no backup needed")
            return True
        
        # Prepare backup data
        backup_data = {
            "collection_name": collection_name,
            "backup_date": datetime.now().isoformat(),
            "points_count": len(points),
            "points": []
        }
        
        for point in points:
            backup_data["points"].append({
                "id": str(point.id),
                "payload": point.payload
            })
        
        # Save to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{collection_name}_backup_{timestamp}.json"
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Backed up {len(points)} points to: {backup_file}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to backup collection '{collection_name}': {e}")
        return False

def clear_collection_points(client, collection_name: str) -> bool:
    """Delete all points from a collection but keep the collection structure"""
    try:
        # Get collection info first
        info = get_collection_info(client, collection_name)
        if not info["exists"]:
            print(f"⚠️  Collection '{collection_name}' does not exist")
            return False
        
        points_count = info["points_count"]
        if points_count == 0:
            print(f"✅ Collection '{collection_name}' is already empty")
            return True
        
        # Delete all points by getting their IDs first
        points_result = client.scroll(
            collection_name=collection_name,
            limit=10000,  # Large limit to get all points
            with_payload=False,
            with_vectors=False
        )
        
        points = points_result[0]
        if points:
            point_ids = [point.id for point in points]
            
            # Delete points in batches
            batch_size = 1000
            deleted_count = 0
            
            for i in range(0, len(point_ids), batch_size):
                batch_ids = point_ids[i:i + batch_size]
                client.delete(
                    collection_name=collection_name,
                    points_selector=batch_ids
                )
                deleted_count += len(batch_ids)
                print(f"🗑️  Deleted {deleted_count}/{len(point_ids)} points...")
            
            print(f"✅ Cleared {deleted_count} points from '{collection_name}'")
            return True
        else:
            print(f"✅ Collection '{collection_name}' is already empty")
            return True
            
    except Exception as e:
        print(f"❌ Failed to clear collection '{collection_name}': {e}")
        return False

def delete_collection(client, collection_name: str) -> bool:
    """Completely delete a collection"""
    try:
        client.delete_collection(collection_name)
        print(f"✅ Deleted collection '{collection_name}'")
        return True
    except Exception as e:
        print(f"❌ Failed to delete collection '{collection_name}': {e}")
        return False

def recreate_collection(client, collection_name: str) -> bool:
    """Delete and recreate a collection with default settings"""
    try:
        # Delete existing collection
        try:
            client.delete_collection(collection_name)
            print(f"🗑️  Deleted existing collection '{collection_name}'")
        except:
            pass  # Collection might not exist
        
        # Recreate collection using setup function
        setup_collections_dynamic()
        print(f"✅ Recreated collection '{collection_name}'")
        return True
    except Exception as e:
        print(f"❌ Failed to recreate collection '{collection_name}': {e}")
        return False

def interactive_clean():
    """Interactive cleaning interface"""
    print_separator("🧹 VECTORDB INTERACTIVE CLEANER")
    
    try:
        # Connect to Qdrant
        client = get_qdrant_client()
        print("✅ Connected to Qdrant successfully")
        
        # Get all collections
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if not collection_names:
            print("❌ No collections found in VectorDB")
            return
        
        print(f"📚 Found collections: {collection_names}")
        
        # Show collection info
        print_separator("📊 COLLECTION STATUS")
        for name in collection_names:
            info = get_collection_info(client, name)
            if info["exists"]:
                print(f"📦 {name}: {info['points_count']} points, {info['vector_size']}D vectors")
            else:
                print(f"❌ {name}: Error - {info['error']}")
        
        while True:
            print_separator("🎛️ CLEANING OPTIONS")
            print("1. 🗑️  Clear all points from specific collection")
            print("2. 💥 Delete entire collection")
            print("3. 🔄 Recreate collection (delete + create)")
            print("4. 🧹 Clear all points from ALL collections")
            print("5. 💀 Delete ALL collections")
            print("6. 🔄 Recreate ALL collections")
            print("7. 💾 Backup collection before cleaning")
            print("8. 📊 Show collection status")
            print("9. ❌ Exit")
            
            try:
                choice = input("\nChoose option (1-9): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Exiting...")
                break
            
            if choice == "1":
                # Clear specific collection
                print("\nAvailable collections:")
                for i, name in enumerate(collection_names, 1):
                    info = get_collection_info(client, name)
                    print(f"  {i}. {name} ({info.get('points_count', 0)} points)")
                
                try:
                    coll_choice = input("Enter collection number: ").strip()
                    coll_idx = int(coll_choice) - 1
                    if 0 <= coll_idx < len(collection_names):
                        collection_name = collection_names[coll_idx]
                        confirm = input(f"⚠️  Clear all points from '{collection_name}'? (y/N): ").strip().lower()
                        if confirm == 'y':
                            clear_collection_points(client, collection_name)
                    else:
                        print("❌ Invalid collection number")
                except (ValueError, IndexError):
                    print("❌ Invalid input")
            
            elif choice == "2":
                # Delete specific collection
                print("\nAvailable collections:")
                for i, name in enumerate(collection_names, 1):
                    print(f"  {i}. {name}")
                
                try:
                    coll_choice = input("Enter collection number: ").strip()
                    coll_idx = int(coll_choice) - 1
                    if 0 <= coll_idx < len(collection_names):
                        collection_name = collection_names[coll_idx]
                        confirm = input(f"⚠️  DELETE entire collection '{collection_name}'? This cannot be undone! (y/N): ").strip().lower()
                        if confirm == 'y':
                            delete_collection(client, collection_name)
                            collection_names.remove(collection_name)
                    else:
                        print("❌ Invalid collection number")
                except (ValueError, IndexError):
                    print("❌ Invalid input")
            
            elif choice == "3":
                # Recreate specific collection
                print("\nAvailable collections:")
                for i, name in enumerate(collection_names, 1):
                    print(f"  {i}. {name}")
                
                try:
                    coll_choice = input("Enter collection number: ").strip()
                    coll_idx = int(coll_choice) - 1
                    if 0 <= coll_idx < len(collection_names):
                        collection_name = collection_names[coll_idx]
                        confirm = input(f"⚠️  RECREATE collection '{collection_name}'? All data will be lost! (y/N): ").strip().lower()
                        if confirm == 'y':
                            recreate_collection(client, collection_name)
                    else:
                        print("❌ Invalid collection number")
                except (ValueError, IndexError):
                    print("❌ Invalid input")
            
            elif choice == "4":
                # Clear all collections
                confirm = input("⚠️  Clear ALL points from ALL collections? (y/N): ").strip().lower()
                if confirm == 'y':
                    for name in collection_names:
                        print(f"\n🧹 Clearing collection: {name}")
                        clear_collection_points(client, name)
                    print("\n✅ All collections cleared!")
            
            elif choice == "5":
                # Delete all collections
                confirm = input("💀 DELETE ALL collections? This cannot be undone! Type 'DELETE ALL' to confirm: ").strip()
                if confirm == 'DELETE ALL':
                    for name in collection_names:
                        print(f"\n💥 Deleting collection: {name}")
                        delete_collection(client, name)
                    collection_names.clear()
                    print("\n💀 All collections deleted!")
                else:
                    print("❌ Deletion cancelled")
            
            elif choice == "6":
                # Recreate all collections
                confirm = input("🔄 RECREATE ALL collections? All data will be lost! Type 'RECREATE ALL' to confirm: ").strip()
                if confirm == 'RECREATE ALL':
                    # Delete all first
                    for name in collection_names:
                        print(f"\n🗑️  Deleting collection: {name}")
                        delete_collection(client, name)
                    
                    # Recreate with setup function
                    print("\n🔄 Recreating collections...")
                    setup_collections_dynamic()
                    
                    # Update collection list
                    collections = client.get_collections()
                    collection_names = [c.name for c in collections.collections]
                    print(f"\n✅ Recreated collections: {collection_names}")
                else:
                    print("❌ Recreation cancelled")
            
            elif choice == "7":
                # Backup collection
                print("\nAvailable collections:")
                for i, name in enumerate(collection_names, 1):
                    info = get_collection_info(client, name)
                    print(f"  {i}. {name} ({info.get('points_count', 0)} points)")
                
                try:
                    coll_choice = input("Enter collection number: ").strip()
                    coll_idx = int(coll_choice) - 1
                    if 0 <= coll_idx < len(collection_names):
                        collection_name = collection_names[coll_idx]
                        backup_dir = project_root / "backups" / "vectordb"
                        backup_collection(client, collection_name, backup_dir)
                    else:
                        print("❌ Invalid collection number")
                except (ValueError, IndexError):
                    print("❌ Invalid input")
            
            elif choice == "8":
                # Show status
                print_separator("📊 CURRENT COLLECTION STATUS")
                collections = client.get_collections()
                collection_names = [c.name for c in collections.collections]
                
                if not collection_names:
                    print("❌ No collections found")
                else:
                    for name in collection_names:
                        info = get_collection_info(client, name)
                        if info["exists"]:
                            print(f"📦 {name}: {info['points_count']} points, {info['vector_size']}D vectors")
                        else:
                            print(f"❌ {name}: Error - {info['error']}")
            
            elif choice == "9":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice")
                
    except Exception as e:
        print(f"❌ Error: {e}")

def quick_clean_all():
    """Quick function to clear all collections without interaction"""
    print_separator("🧹 QUICK CLEAN ALL COLLECTIONS")
    
    try:
        client = get_qdrant_client()
        print("✅ Connected to Qdrant successfully")
        
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if not collection_names:
            print("❌ No collections found")
            return
        
        print(f"📚 Found collections: {collection_names}")
        
        for name in collection_names:
            print(f"\n🧹 Clearing collection: {name}")
            clear_collection_points(client, name)
        
        print("\n✅ All collections cleaned successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean VectorDB collections')
    parser.add_argument('--quick-clean', action='store_true', help='Clear all collections without interaction')
    parser.add_argument('--recreate-all', action='store_true', help='Delete and recreate all collections')
    
    args = parser.parse_args()
    
    if args.quick_clean:
        quick_clean_all()
    elif args.recreate_all:
        print_separator("🔄 RECREATE ALL COLLECTIONS")
        try:
            client = get_qdrant_client()
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            # Delete all
            for name in collection_names:
                print(f"🗑️  Deleting collection: {name}")
                delete_collection(client, name)
            
            # Recreate
            print("🔄 Recreating collections...")
            setup_collections_dynamic()
            print("✅ All collections recreated!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        interactive_clean()

if __name__ == "__main__":
    main()
