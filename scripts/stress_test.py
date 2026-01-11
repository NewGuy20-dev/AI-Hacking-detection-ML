#!/usr/bin/env python3
"""Stress test with truly novel benign samples to find real FP rate."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

# Novel benign samples NOT in training data
NOVEL_BENIGN = [
    # Emails with unusual patterns
    "support.team.usa@company.io",
    "no-reply@newsletter.co.uk",
    "admin.backup.2024@server.net",
    
    # Social media style
    "@username mentioned you",
    "#trending #viral #fyp",
    "RT @user: great post!",
    "DM me for details",
    
    # Emojis and unicode
    "🎉 Congratulations! 🎊",
    "Check this out 👀",
    "100% satisfied 💯",
    "Love this ❤️",
    
    # Timestamps and dates
    "Meeting at 2024-12-22 15:30",
    "Updated: 12/22/2024 3:30 PM",
    "Event starts @ 9am EST",
    
    # Currency and numbers
    "Total: $1,234.56",
    "Price: €99.99",
    "Balance: £500.00",
    "Discount: -20%",
    
    # Technical but benign
    "API response: 200 OK",
    "Status: SUCCESS",
    "Error: None",
    "Connection timeout: 30s",
    "Memory usage: 512MB",
    "CPU: 45%",
    
    # Paths with spaces
    "C:\\Program Files\\My App\\config.ini",
    "/home/john doe/my documents/file.txt",
    
    # Queries that look suspicious but aren't
    "Search: how to fix SQL error",
    "Query: best script writing software",
    "Looking for: DROP shipping suppliers",
    
    # Code-like but conversational
    "if you need help, let me know",
    "select your preferred option below",
    "join our community today",
    "update your preferences anytime",
    
    # Mixed content
    "Order #12345 - Status: Shipped",
    "Invoice INV-2024-001 attached",
    "Ticket #ABC123 resolved",
    
    # Common phrases with special chars
    "Q&A session tomorrow",
    "R&D department update",
    "Terms & Conditions apply",
    "50/50 split",
    "24/7 support available",
    
    # Filenames
    "report_2024_final_v2.xlsx",
    "photo_IMG_1234.jpg",
    "backup-2024-12-22.tar.gz",
    
    # Markdown-like
    "**Important:** Read this first",
    "*Note:* This is optional",
    "# Heading 1",
    "- Item 1\n- Item 2",
    
    # JSON-like strings
    "config: {debug: true}",
    "options: [a, b, c]",
    "user.name = 'John'",
    
    # Regex-like patterns (benign)
    "Pattern: [a-z]+",
    "Format: YYYY-MM-DD",
    "Mask: ***-**-1234",
    
    # Log entries
    "[INFO] Application started",
    "[2024-12-22] User logged in",
    "DEBUG: Processing request",
    
    # Version strings
    "Version 2.0.1-beta",
    "v1.2.3 released",
    "Build #4567",
    
    # Network addresses (benign)
    "Server: 192.168.1.100:8080",
    "Localhost: 127.0.0.1",
    "DNS: 8.8.8.8",
    
    # Base64-like (but not actual exploits)
    "Token: abc123xyz789",
    "Key: ABCD-EFGH-IJKL",
    "Code: X1Y2Z3",
    
    # HTML entities
    "&amp; &lt; &gt; &quot;",
    "Copyright &copy; 2024",
    
    # Escaped characters
    "Path: C:\\\\Users\\\\Admin",
    "Quote: \\"Hello\\"",
    "Tab:\\tNext",
]

# Known malicious patterns (should be detected)
KNOWN_MALICIOUS = [
    # SQL Injection
    "' OR '1'='1",
    "'; DROP TABLE users--",
    "' UNION SELECT * FROM users--",
    "1' AND '1'='1",
    "admin'--",
    "' OR 1=1--",
    
    # XSS
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert(1)>",
    "<svg onload=alert(1)>",
    "javascript:alert(document.cookie)",
    "<body onload=alert(1)>",
    
    # Command Injection
    "| cat /etc/passwd",
    "; ls -la",
    "$(whoami)",
    "`id`",
    "& ping -c 10 127.0.0.1",
    "; rm -rf /",
    
    # Template Injection
    "{{7*7}}",
    "${7*7}",
    "#{7*7}",
    "<%= system('id') %>",
    
    # Path Traversal
    "../../../etc/passwd",
    "....//....//etc/passwd",
    "%2e%2e%2f%2e%2e%2fetc/passwd",
    
    # LDAP Injection
    "*)(uid=*))(|(uid=*",
    "admin)(&)",
    
    # XML/XXE
    "<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>",
]


def main():
    base_path = Path(__file__).parent.parent
    
    print("=" * 60)
    print(" STRESS TEST - Novel Samples")
    print("=" * 60)
    
    # Load model
    from hybrid_predictor import HybridPredictor
    predictor = HybridPredictor(str(base_path / "models"))
    predictor.load_models()
    
    # Test benign samples
    print("\n--- Testing Novel Benign Samples ---")
    benign_scores = predictor.predict_payload(NOVEL_BENIGN).flatten()
    
    fp_count = 0
    print("\nFalse Positives (score > 0.5):")
    for text, score in zip(NOVEL_BENIGN, benign_scores):
        if score > 0.5:
            fp_count += 1
            print(f"  FP: '{text[:50]}...' -> {score:.1%}")
    
    fp_rate = fp_count / len(NOVEL_BENIGN)
    print(f"\nBenign: {len(NOVEL_BENIGN) - fp_count}/{len(NOVEL_BENIGN)} correct")
    print(f"FP Rate: {fp_rate:.1%} ({fp_count} false positives)")
    
    # Test malicious samples
    print("\n--- Testing Known Malicious Samples ---")
    mal_scores = predictor.predict_payload(KNOWN_MALICIOUS).flatten()
    
    fn_count = 0
    print("\nFalse Negatives (score < 0.5):")
    for text, score in zip(KNOWN_MALICIOUS, mal_scores):
        if score < 0.5:
            fn_count += 1
            print(f"  FN: '{text[:50]}...' -> {score:.1%}")
    
    recall = (len(KNOWN_MALICIOUS) - fn_count) / len(KNOWN_MALICIOUS)
    print(f"\nMalicious: {len(KNOWN_MALICIOUS) - fn_count}/{len(KNOWN_MALICIOUS)} detected")
    print(f"Recall: {recall:.1%}")
    
    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"FP Rate: {fp_rate:.1%}")
    print(f"Recall: {recall:.1%}")
    
    total = len(NOVEL_BENIGN) + len(KNOWN_MALICIOUS)
    correct = (len(NOVEL_BENIGN) - fp_count) + (len(KNOWN_MALICIOUS) - fn_count)
    print(f"Accuracy: {correct/total:.1%}")


if __name__ == "__main__":
    main()
