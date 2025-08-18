#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test RFP Tracking System
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from qdrant.rfp_tracker import get_rfp_tracker
from qdrant.indexer import index_completed_rfp

def test_rfp_tracking():
    """Test the RFP tracking and numbering system"""
    print("🧪 Testing RFP Tracking System")
    print("=" * 50)
    
    # Get tracker and show initial stats
    tracker = get_rfp_tracker()
    initial_stats = tracker.get_stats()
    
    print("📊 Initial Stats:")
    print(f"  Current RFP: {initial_stats['current_rfp_number']}")
    print(f"  Total processed: {initial_stats['total_rfps_processed']}")
    print(f"  Cleanup enabled: {initial_stats['cleanup_enabled']}")
    print(f"  Max age diff: {initial_stats['max_age_difference']}")
    
    # Test getting next RFP number
    print("\n🔢 Testing RFP numbering:")
    next_rfp_1 = tracker.get_next_rfp_number()
    print(f"  Next RFP: {next_rfp_1}")
    
    next_rfp_2 = tracker.get_next_rfp_number()
    print(f"  Next RFP: {next_rfp_2}")
    
    current = tracker.get_current_rfp_number()
    print(f"  Current RFP: {current}")
    
    # Test configuration changes
    print("\n⚙️ Testing configuration:")
    tracker.set_max_age_difference(15)
    tracker.enable_cleanup(True)
    
    # Show final stats
    final_stats = tracker.get_stats()
    print("\n📊 Final Stats:")
    print(f"  Current RFP: {final_stats['current_rfp_number']}")
    print(f"  Total processed: {final_stats['total_rfps_processed']}")
    print(f"  Cleanup enabled: {final_stats['cleanup_enabled']}")
    print(f"  Max age diff: {final_stats['max_age_difference']}")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_rfp_tracking()
