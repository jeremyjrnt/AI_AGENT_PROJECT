#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RFP Management CLI
Command-line tool for managing RFP tracking and cleanup
"""

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from qdrant.rfp_tracker import get_rfp_tracker
from qdrant.client import get_qdrant_client, RFP_QA_COLLECTION

def show_stats():
    """Display RFP tracking statistics"""
    tracker = get_rfp_tracker()
    stats = tracker.get_stats()
    
    print("=" * 60)
    print("📊 RFP TRACKING STATISTICS")
    print("=" * 60)
    print(f"🔢 Current RFP Number: {stats['current_rfp_number']}")
    print(f"📈 Total RFPs Processed: {stats['total_rfps_processed']}")
    print(f"🧹 Cleanup Enabled: {'✅' if stats['cleanup_enabled'] else '❌'}")
    print(f"⏰ Max Age Difference: {stats['max_age_difference']}")
    print(f"🕒 Last Updated: {stats['last_updated']}")
    print(f"📁 State File: {stats['state_file']}")
    print("=" * 60)

def force_cleanup():
    """Force cleanup of old RFP documents"""
    tracker = get_rfp_tracker()
    print("🧹 Starting forced cleanup of old RFP documents...")
    
    cleanup_count = tracker.cleanup_old_rfps(force=True)
    
    if cleanup_count > 0:
        print(f"✅ Successfully cleaned up {cleanup_count} old documents")
    else:
        print("📝 No documents required cleanup")

def set_max_age(age: int):
    """Set maximum age difference for cleanup"""
    tracker = get_rfp_tracker()
    tracker.set_max_age_difference(age)
    print(f"✅ Max age difference set to {age}")

def toggle_cleanup(enable: bool):
    """Enable or disable automatic cleanup"""
    tracker = get_rfp_tracker()
    tracker.enable_cleanup(enable)
    status = "enabled" if enable else "disabled"
    print(f"✅ Automatic cleanup {status}")

def reset_counter(value: int):
    """Reset RFP counter"""
    tracker = get_rfp_tracker()
    
    # Confirmation
    current = tracker.get_current_rfp_number()
    print(f"⚠️  WARNING: This will reset RFP counter from {current} to {value}")
    confirm = input("Are you sure? (y/N): ").strip().lower()
    
    if confirm == 'y':
        tracker.reset_counter(value)
        print(f"✅ RFP counter reset to {value}")
    else:
        print("❌ Reset cancelled")

def inspect_collection():
    """Inspect RFP collection for age distribution"""
    try:
        client = get_qdrant_client()
        tracker = get_rfp_tracker()
        current_rfp = tracker.get_current_rfp_number()
        
        # Get all points
        scroll_result = client.scroll(
            collection_name=RFP_QA_COLLECTION,
            limit=10000,
            with_payload=True
        )
        
        rfp_distribution = {}
        total_points = len(scroll_result[0])
        
        for point in scroll_result[0]:
            payload = point.payload or {}
            rfp_number = payload.get("rfp_number", "unknown")
            
            if rfp_number not in rfp_distribution:
                rfp_distribution[rfp_number] = 0
            rfp_distribution[rfp_number] += 1
        
        print("=" * 60)
        print("📊 RFP COLLECTION ANALYSIS")
        print("=" * 60)
        print(f"📈 Total Documents: {total_points}")
        print(f"🔢 Current RFP: {current_rfp}")
        print(f"📊 RFP Distribution:")
        
        for rfp_num in sorted(rfp_distribution.keys()):
            count = rfp_distribution[rfp_num]
            age = current_rfp - rfp_num if isinstance(rfp_num, int) else "N/A"
            age_status = ""
            if isinstance(age, int):
                if age > 20:
                    age_status = " ⚠️  OLD"
                elif age > 10:
                    age_status = " ⚡ AGING"
            
            print(f"  RFP #{rfp_num}: {count} docs (age: {age}){age_status}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error inspecting collection: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="RFP Management CLI - Manage RFP tracking and cleanup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rfp_manager_cli.py --stats                    Show statistics
  python rfp_manager_cli.py --cleanup                  Force cleanup old RFPs
  python rfp_manager_cli.py --set-max-age 25           Set max age to 25
  python rfp_manager_cli.py --enable-cleanup           Enable auto cleanup
  python rfp_manager_cli.py --disable-cleanup          Disable auto cleanup
  python rfp_manager_cli.py --reset-counter 50         Reset counter to 50
  python rfp_manager_cli.py --inspect                  Inspect collection
        """
    )
    
    parser.add_argument('--stats', action='store_true', help='Show RFP tracking statistics')
    parser.add_argument('--cleanup', action='store_true', help='Force cleanup of old RFP documents')
    parser.add_argument('--set-max-age', type=int, metavar='AGE', help='Set maximum age difference for cleanup')
    parser.add_argument('--enable-cleanup', action='store_true', help='Enable automatic cleanup')
    parser.add_argument('--disable-cleanup', action='store_true', help='Disable automatic cleanup')
    parser.add_argument('--reset-counter', type=int, metavar='VALUE', help='Reset RFP counter to specified value')
    parser.add_argument('--inspect', action='store_true', help='Inspect RFP collection age distribution')
    
    args = parser.parse_args()
    
    # Default to showing stats if no arguments
    if not any(vars(args).values()):
        show_stats()
        return
    
    if args.stats:
        show_stats()
    
    if args.cleanup:
        force_cleanup()
    
    if args.set_max_age is not None:
        set_max_age(args.set_max_age)
    
    if args.enable_cleanup:
        toggle_cleanup(True)
    
    if args.disable_cleanup:
        toggle_cleanup(False)
    
    if args.reset_counter is not None:
        reset_counter(args.reset_counter)
    
    if args.inspect:
        inspect_collection()

if __name__ == "__main__":
    main()
