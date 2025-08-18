#!/usr/bin/env python3
"""
Collection Inspector - VectorDB Random Points Viewer
Inspects collections and shows 10 raw data points randomly selected
"""

import sys
from pathlib import Path
import json
import random
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from qdrant.client import get_qdrant_client
import settings

def print_separator(title: str, char: str = "=", length: int = 100):
    """Print a formatted separator with title"""
    print(f"\n{char * length}")
    print(f"{title:^{length}}")
    print(f"{char * length}")

def print_raw_point(point, point_number: int, collection_name: str):
    """Print complete raw data for a single point"""
    print(f"\n{'=' * 100}")
    print(f"🔍 RANDOM POINT #{point_number} - COLLECTION: {collection_name.upper()}")
    print(f"ID: {point.id}")
    print(f"{'=' * 100}")
    
    # Raw Vector (first 10 dimensions only to avoid clutter)
    if hasattr(point, 'vector') and point.vector:
        vector_preview = point.vector[:10] if len(point.vector) > 10 else point.vector
        print(f"\n🔢 VECTOR (first 10 dims): {vector_preview}...")
        print(f"   📐 Total dimensions: {len(point.vector)}")
    
    # Complete Raw Payload as JSON
    print(f"\n📦 RAW PAYLOAD (JSON FORMAT):")
    print("─" * 60)
    
    if hasattr(point, 'payload') and point.payload:
        try:
            # Format as JSON with proper indentation
            formatted_payload = json.dumps(point.payload, indent=2, ensure_ascii=False)
            print(formatted_payload)
        except Exception as e:
            print(f"❌ Error formatting JSON: {e}")
            # Fallback to direct printing
            for key, value in point.payload.items():
                print(f'"{key}": {json.dumps(value, ensure_ascii=False)}')
    else:
        print("   No payload data")
    
    print("─" * 60)

def get_random_points(client, collection_name: str, num_points: int = 10):
    """Get random points from a collection"""
    try:
        # Get collection info first
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count
        
        print(f"📊 Collection '{collection_name}' has {total_points} total points")
        
        if total_points == 0:
            print("❌ Collection is empty")
            return []
        
        # Use scroll to get more points than we need, then randomly select
        scroll_limit = min(100, total_points)  # Get up to 100 points to choose from
        
        # Get points with scroll
        points_result = client.scroll(
            collection_name=collection_name,
            limit=scroll_limit,
            with_payload=True,
            with_vectors=True
        )
        
        points = points_result[0] if points_result else []
        
        if not points:
            print("❌ No points retrieved from collection")
            return []
        
        # Randomly select points
        actual_num_points = min(num_points, len(points))
        random_points = random.sample(points, actual_num_points)
        
        print(f"🎲 Randomly selected {actual_num_points} points from {len(points)} available")
        
        return random_points
        
    except Exception as e:
        print(f"❌ Error getting random points from {collection_name}: {e}")
        return []

def inspect_collection_random(client, collection_name: str, num_points: int = 10):
    """Inspect a collection and show random points"""
    print_separator(f"🎲 RANDOM INSPECTION: {collection_name.upper()}")
    
    try:
        # Get collection statistics
        collection_info = client.get_collection(collection_name)
        print(f"📈 Collection Statistics:")
        print(f"   📊 Total Points: {collection_info.points_count}")
        print(f"   📐 Vector Size: {collection_info.config.params.vectors.size}")
        print(f"   📏 Distance Metric: {collection_info.config.params.vectors.distance}")
        
        # Get random points
        random_points = get_random_points(client, collection_name, num_points)
        
        if not random_points:
            return
        
        print(f"\n🔍 Showing {len(random_points)} random points:")
        
        # Display each random point
        for i, point in enumerate(random_points, 1):
            print_raw_point(point, i, collection_name)
            
    except Exception as e:
        print(f"❌ Error inspecting collection {collection_name}: {e}")

def inspect_all_collections():
    """Inspect all collections in the VectorDB"""
    print_separator("🔍 VECTORDB COLLECTIONS INSPECTOR", "=", 120)
    print("🎲 Showing 10 random raw data points from each collection")
    
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
        
        print(f"📚 Found {len(collection_names)} collections: {collection_names}")
        
        # Inspect each collection
        for collection_name in collection_names:
            inspect_collection_random(client, collection_name, num_points=10)
        
        print_separator("✅ INSPECTION COMPLETE", "=", 120)
        
    except Exception as e:
        print(f"❌ Error connecting to VectorDB: {e}")

def inspect_specific_collection(collection_name: str, num_points: int = 10):
    """Inspect a specific collection"""
    print_separator(f"🎯 SPECIFIC COLLECTION INSPECTOR: {collection_name.upper()}", "=", 120)
    
    try:
        # Connect to Qdrant
        client = get_qdrant_client()
        print("✅ Connected to Qdrant successfully")
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if collection_name not in collection_names:
            print(f"❌ Collection '{collection_name}' not found")
            print(f"Available collections: {collection_names}")
            return
        
        # Inspect the specific collection
        inspect_collection_random(client, collection_name, num_points)
        
    except Exception as e:
        print(f"❌ Error inspecting collection {collection_name}: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect VectorDB collections and show random points')
    parser.add_argument('--collection', '-c', type=str, help='Specific collection name to inspect')
    parser.add_argument('--points', '-p', type=int, default=10, help='Number of random points to show (default: 10)')
    parser.add_argument('--all', '-a', action='store_true', help='Inspect all collections')
    
    args = parser.parse_args()
    
    if args.collection:
        inspect_specific_collection(args.collection, args.points)
    else:
        # Default: inspect all collections
        inspect_all_collections()

if __name__ == "__main__":
    main()
