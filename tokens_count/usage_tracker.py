#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local usage tracker for embeddings
"""

import json
import os
from datetime import datetime
from pathlib import Path

class LocalUsageTracker:
    def __init__(self, usage_file="usage_log.json"):
        # Ne pas sauvegarder de façon persistante - données en mémoire seulement
        self.usage_data = {"sessions": [], "total_tokens": 0, "total_requests": 0}
    
    def load_usage(self):
        """Les données ne sont pas chargées depuis un fichier - mémoire seulement"""
        pass
    
    def save_usage(self):
        """Les données ne sont pas sauvegardées - mémoire seulement"""
        pass
    
    def log_embedding_request(self, texts, provider="azure_openai", model="text-embedding-3-large"):
        """Log an embedding request"""
        if isinstance(texts, str):
            texts = [texts]
        
        total_chars = sum(len(text) for text in texts)
        estimated_tokens = total_chars // 4  # Rough estimation
        
        # Price per 1K tokens
        pricing = {
            "azure_openai": {
                "text-embedding-3-large": 0.00013,
                "text-embedding-3-small": 0.00002,
                "team11-embedding": 0.00013  # Assumed similar to text-embedding-3-large
            },
            "openai": {
                "text-embedding-3-large": 0.00013,
                "text-embedding-3-small": 0.00002,
                "text-embedding-ada-002": 0.00010
            },
            "huggingface": {
                "all-MiniLM-L6-v2": 0.0  # Free
            }
        }
        
        cost_per_1k = pricing.get(provider, {}).get(model, 0.00013)
        estimated_cost = (estimated_tokens / 1000) * cost_per_1k
        
        session = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "num_texts": len(texts),
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens,
            "estimated_cost_usd": round(estimated_cost, 8),
            "texts_preview": texts[:2] if len(texts) <= 5 else texts[:2] + ["..."]
        }
        
        # Add to history (en mémoire seulement)
        self.usage_data["sessions"].append(session)
        self.usage_data["total_tokens"] += estimated_tokens
        self.usage_data["total_requests"] += 1
        
        # Keep only last 100 sessions
        if len(self.usage_data["sessions"]) > 100:
            self.usage_data["sessions"] = self.usage_data["sessions"][-100:]
        
        # Pas de sauvegarde persistante
        return session
    
    def get_usage_summary(self, days=7):
        """Get usage summary"""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_sessions = [
            s for s in self.usage_data["sessions"]
            if datetime.fromisoformat(s["timestamp"]) >= cutoff_date
        ]
        
        total_cost = sum(s.get("estimated_cost_usd", 0) for s in recent_sessions)
        total_tokens = sum(s.get("estimated_tokens", 0) for s in recent_sessions)
        total_requests = len(recent_sessions)
        
        providers = {}
        for session in recent_sessions:
            provider = session.get("provider", "unknown")
            if provider not in providers:
                providers[provider] = {"requests": 0, "tokens": 0, "cost": 0}
            providers[provider]["requests"] += 1
            providers[provider]["tokens"] += session.get("estimated_tokens", 0)
            providers[provider]["cost"] += session.get("estimated_cost_usd", 0)
        
        return {
            "period_days": days,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "providers": providers,
            "recent_sessions": recent_sessions[-5:]  # Last 5
        }
    
    def print_summary(self, days=7):
        """Print formatted summary"""
        summary = self.get_usage_summary(days)
        
        print(f"📊 Usage Summary ({days} last days)")
        print("=" * 50)
        print(f"Total requests: {summary['total_requests']}")
        print(f"Estimated tokens: {summary['total_tokens']:,}")
        print(f"Estimated cost: ${summary['total_cost_usd']:.6f} USD")
        print()
        
        print("By provider:")
        for provider, stats in summary['providers'].items():
            cost = stats['cost']
            print(f"  {provider}: {stats['requests']} requests, {stats['tokens']:,} tokens, ${cost:.6f}")
        print()
        
        if summary['recent_sessions']:
            print("Recent requests:")
            for session in summary['recent_sessions']:
                timestamp = datetime.fromisoformat(session['timestamp'])
                time_str = timestamp.strftime('%H:%M:%S')
                provider = session.get('provider', 'N/A')
                tokens = session.get('estimated_tokens', 0)
                cost = session.get('estimated_cost_usd', 0)
                preview = session.get('texts_preview', [''])[0][:30] + "..." if session.get('texts_preview') else "N/A"
                print(f"  {time_str} | {provider} | {tokens} tokens | ${cost:.6f} | '{preview}'")

# Simple display - données en mémoire seulement (non persistantes)
if __name__ == "__main__":
    print("📊 Usage Tracker - Mode Mémoire Seulement")
    print("=" * 45)
    print("ℹ️  Les données de tracking ne survivent pas aux redémarrages")
    print("ℹ️  Seules les données de la session courante sont trackées")
    print()
    
    # Créer un tracker temporaire pour la démonstration
    tracker = LocalUsageTracker()
    
    if not tracker.usage_data["sessions"]:
        print("📊 Aucun appel d'embedding enregistré dans cette session")
        print()
        print("📝 Données trackées lors des appels :")
        print("   - timestamp")
        print("   - provider") 
        print("   - tokens utilisés")
        print("   - coût estimé")
        print("   - nombre de textes traités")
        print()
        print("💡 Les données sont perdues à chaque redémarrage du système")
    else:
        # Calculate comprehensive statistics
        total_sessions = len(tracker.usage_data["sessions"])
        total_tokens = sum(s.get('estimated_tokens', 0) for s in tracker.usage_data["sessions"])
        total_cost = sum(s.get('estimated_cost_usd', 0) for s in tracker.usage_data["sessions"])
        
        # Build output text for both display and file saving
        output_lines = []
        output_lines.append("📊 Embeddings Monitoring")
        output_lines.append("=" * 40)
        output_lines.append("")
        output_lines.append("📈 TOTAL USAGE SINCE BEGINNING:")
        output_lines.append(f"🎯 Total tokens: {total_tokens:,}")
        output_lines.append(f"💰 Total estimated cost: ${total_cost:.6f} USD")
        output_lines.append("")
        
        # Show last 5 calls
        recent_sessions = tracker.usage_data["sessions"][-5:]
        output_lines.append(f"🕐 Last {len(recent_sessions)} calls:")
        
        total_recent_cost = 0
        for session in recent_sessions:
            timestamp = datetime.fromisoformat(session['timestamp'])
            time_str = timestamp.strftime('%d/%m %H:%M:%S')
            provider = session.get('provider', 'N/A')
            tokens = session.get('estimated_tokens', 0)
            cost = session.get('estimated_cost_usd', 0)
            total_recent_cost += cost
            num_texts = session.get('num_texts', 1)
            preview = session.get('texts_preview', [''])[0][:35] + "..." if session.get('texts_preview') else "N/A"
            
            output_lines.append(f"  {time_str} | {provider:12} | {tokens:3}t | ${cost:.6f} | {num_texts} text{'s' if num_texts > 1 else ''} | '{preview}'")
        
        output_lines.append("")
        output_lines.append(f"💰 Estimated Cost on last 5 requests: ${total_recent_cost:.6f} USD")
        
        # Print to console seulement (données non persistantes)
        output_text = "\n".join(output_lines)
        print(output_text)
        
        print(f"\n💡 Note: Ces données sont uniquement en mémoire et seront perdues au redémarrage")
