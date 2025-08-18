#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RFP Tracking System
- Manages RFP numbering and automatic cleanup
- Maintains current RFP number and handles age-based cleanup
"""

from __future__ import annotations
from typing import Dict, Any, List
import json
import os
from pathlib import Path
from datetime import datetime
from .client import get_qdrant_client, RFP_QA_COLLECTION

# File to store RFP tracking state
RFP_STATE_FILE = Path(__file__).parent.parent / "rfp_state.json"

class RFPTracker:
    """Manages RFP numbering and automatic cleanup of old RFP data"""
    
    def __init__(self):
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load RFP state from file"""
        if RFP_STATE_FILE.exists():
            try:
                with open(RFP_STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading RFP state: {e}")
        
        # Default state
        return {
            "current_rfp_number": 0,
            "last_updated": datetime.now().isoformat(),
            "total_rfps_processed": 0,
            "cleanup_enabled": True,
            "max_age_difference": 20
        }
    
    def _save_state(self):
        """Save current state to file"""
        try:
            self.state["last_updated"] = datetime.now().isoformat()
            with open(RFP_STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving RFP state: {e}")
    
    def get_next_rfp_number(self) -> int:
        """Get the next RFP number and increment counter"""
        self.state["current_rfp_number"] += 1
        self.state["total_rfps_processed"] += 1
        self._save_state()
        return self.state["current_rfp_number"]
    
    def get_current_rfp_number(self) -> int:
        """Get current RFP number without incrementing"""
        return self.state["current_rfp_number"]
    
    def cleanup_old_rfps(self, force: bool = False) -> int:
        """
        Remove RFP documents that are too old compared to current RFP
        
        Args:
            force: Force cleanup even if disabled in settings
            
        Returns:
            int: Number of documents cleaned up
        """
        if not self.state.get("cleanup_enabled", True) and not force:
            print("📝 Cleanup disabled in settings")
            return 0
        
        current_rfp = self.state["current_rfp_number"]
        max_age_diff = self.state.get("max_age_difference", 20)
        
        if current_rfp <= max_age_diff:
            print(f"📝 No cleanup needed - current RFP ({current_rfp}) <= max age difference ({max_age_diff})")
            return 0
        
        min_rfp_to_keep = current_rfp - max_age_diff
        
        try:
            client = get_qdrant_client()
            
            # Get all points from collection
            scroll_result = client.scroll(
                collection_name=RFP_QA_COLLECTION,
                limit=10000,  # Adjust based on your collection size
                with_payload=True
            )
            
            points_to_delete = []
            
            for point in scroll_result[0]:  # scroll_result is (points, next_page_offset)
                payload = point.payload or {}
                rfp_number = payload.get("rfp_number")
                
                if rfp_number is not None and rfp_number < min_rfp_to_keep:
                    points_to_delete.append(point.id)
            
            if points_to_delete:
                client.delete(
                    collection_name=RFP_QA_COLLECTION,
                    points_selector=points_to_delete
                )
                print(f"🧹 Cleaned up {len(points_to_delete)} old RFP documents (RFP < {min_rfp_to_keep})")
                return len(points_to_delete)
            else:
                print("📝 No old documents found to cleanup")
                return 0
                
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
            return 0
    
    def set_max_age_difference(self, max_age: int):
        """Update maximum age difference for cleanup"""
        self.state["max_age_difference"] = max_age
        self._save_state()
        print(f"📝 Updated max age difference to {max_age}")
    
    def enable_cleanup(self, enabled: bool = True):
        """Enable or disable automatic cleanup"""
        self.state["cleanup_enabled"] = enabled
        self._save_state()
        status = "enabled" if enabled else "disabled"
        print(f"📝 Automatic cleanup {status}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current RFP tracking statistics"""
        return {
            "current_rfp_number": self.state["current_rfp_number"],
            "total_rfps_processed": self.state["total_rfps_processed"],
            "cleanup_enabled": self.state.get("cleanup_enabled", True),
            "max_age_difference": self.state.get("max_age_difference", 20),
            "last_updated": self.state.get("last_updated"),
            "state_file": str(RFP_STATE_FILE)
        }
    
    def reset_counter(self, new_value: int = 0):
        """Reset RFP counter (use with caution)"""
        old_value = self.state["current_rfp_number"]
        self.state["current_rfp_number"] = new_value
        self._save_state()
        print(f"🔄 Reset RFP counter from {old_value} to {new_value}")

# Global tracker instance
rfp_tracker = RFPTracker()

def get_rfp_tracker() -> RFPTracker:
    """Get the global RFP tracker instance"""
    return rfp_tracker
