"""Analyst checklist generator for security alerts."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class ChecklistItem:
    """Single checklist item."""
    task: str
    category: str  # verification, investigation, response, documentation
    priority: int  # 1-5, 1 being highest
    completed: bool = False


@dataclass
class AnalystChecklist:
    """Complete analyst checklist for an alert."""
    alert_id: str
    attack_type: str
    severity: str
    items: List[ChecklistItem] = field(default_factory=list)
    generated_at: str = ""
    
    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "attack_type": self.attack_type,
            "severity": self.severity,
            "generated_at": self.generated_at,
            "items": [{"task": i.task, "category": i.category, 
                      "priority": i.priority, "completed": i.completed} for i in self.items]
        }
    
    def to_markdown(self) -> str:
        """Generate markdown checklist."""
        lines = [
            f"# Alert Checklist: {self.alert_id}",
            f"**Attack Type:** {self.attack_type}",
            f"**Severity:** {self.severity}",
            f"**Generated:** {self.generated_at}",
            ""
        ]
        
        categories = {}
        for item in self.items:
            if item.category not in categories:
                categories[item.category] = []
            categories[item.category].append(item)
        
        for cat, items in categories.items():
            lines.append(f"## {cat.title()}")
            for item in sorted(items, key=lambda x: x.priority):
                check = "x" if item.completed else " "
                lines.append(f"- [{check}] P{item.priority}: {item.task}")
            lines.append("")
        
        return "\n".join(lines)


class ChecklistGenerator:
    """Generate analyst checklists based on alert type."""
    
    # Base verification steps for all alerts
    BASE_VERIFICATION = [
        ("Verify alert is not a known false positive", 1),
        ("Check source IP/user reputation", 1),
        ("Review timestamp and context", 2),
        ("Confirm affected asset/system", 2),
    ]
    
    # Attack-specific checklists
    ATTACK_CHECKLISTS = {
        "sql_injection": {
            "verification": [
                ("Verify SQL syntax in payload", 1),
                ("Check if payload reached database", 1),
                ("Review application input validation", 2),
            ],
            "investigation": [
                ("Search for similar requests from same source", 1),
                ("Check database query logs", 1),
                ("Review application error logs", 2),
                ("Identify affected database tables", 2),
            ],
            "response": [
                ("Block source IP if confirmed malicious", 1),
                ("Reset affected user sessions", 2),
                ("Patch vulnerable endpoint", 2),
            ],
            "documentation": [
                ("Document attack vector", 3),
                ("Record IOCs (IPs, payloads)", 3),
                ("Update detection rules if needed", 4),
            ]
        },
        "xss": {
            "verification": [
                ("Verify XSS payload syntax", 1),
                ("Check if payload was stored", 1),
                ("Test for reflected XSS", 2),
            ],
            "investigation": [
                ("Identify injection point", 1),
                ("Check for stored XSS in database", 1),
                ("Review affected pages", 2),
                ("Check for session hijacking attempts", 2),
            ],
            "response": [
                ("Sanitize stored malicious content", 1),
                ("Implement output encoding", 2),
                ("Invalidate potentially compromised sessions", 2),
            ],
            "documentation": [
                ("Document vulnerable endpoint", 3),
                ("Record payload variations", 3),
            ]
        },
        "command_injection": {
            "verification": [
                ("Verify command syntax in payload", 1),
                ("Check if command was executed", 1),
                ("Review system call logs", 1),
            ],
            "investigation": [
                ("Check process execution logs", 1),
                ("Review file system changes", 1),
                ("Look for persistence mechanisms", 1),
                ("Check for lateral movement", 2),
            ],
            "response": [
                ("Isolate affected system", 1),
                ("Kill malicious processes", 1),
                ("Block source IP", 1),
                ("Restore from clean backup if needed", 2),
            ],
            "documentation": [
                ("Document full attack chain", 2),
                ("Preserve forensic evidence", 2),
                ("Report to incident response", 2),
            ]
        },
        "malicious_url": {
            "verification": [
                ("Check URL against threat intel", 1),
                ("Verify domain registration details", 2),
                ("Test URL in sandbox", 2),
            ],
            "investigation": [
                ("Identify users who clicked", 1),
                ("Check for malware downloads", 1),
                ("Review proxy/firewall logs", 2),
            ],
            "response": [
                ("Block URL at proxy/firewall", 1),
                ("Notify affected users", 2),
                ("Scan endpoints for malware", 2),
            ],
            "documentation": [
                ("Add URL to blocklist", 3),
                ("Document campaign indicators", 3),
            ]
        },
        "network_attack": {
            "verification": [
                ("Verify attack pattern in traffic", 1),
                ("Check for service impact", 1),
                ("Confirm source IP is external", 2),
            ],
            "investigation": [
                ("Analyze traffic patterns", 1),
                ("Check for data exfiltration", 1),
                ("Review firewall logs", 2),
                ("Identify attack vector", 2),
            ],
            "response": [
                ("Block attacking IPs", 1),
                ("Enable rate limiting", 2),
                ("Notify upstream provider if DDoS", 2),
            ],
            "documentation": [
                ("Document attack timeline", 3),
                ("Record traffic statistics", 3),
            ]
        }
    }
    
    # Default checklist for unknown attack types
    DEFAULT_CHECKLIST = {
        "verification": [
            ("Verify detection accuracy", 1),
            ("Check for false positive indicators", 2),
        ],
        "investigation": [
            ("Gather additional context", 1),
            ("Search for related events", 2),
        ],
        "response": [
            ("Assess risk level", 1),
            ("Determine appropriate response", 2),
        ],
        "documentation": [
            ("Document findings", 3),
        ]
    }
    
    def generate(self, alert_id: str, attack_type: str, severity: str,
                confidence: float = 0.0) -> AnalystChecklist:
        """Generate checklist for an alert."""
        checklist = AnalystChecklist(
            alert_id=alert_id,
            attack_type=attack_type,
            severity=severity
        )
        
        # Add base verification steps
        for task, priority in self.BASE_VERIFICATION:
            checklist.items.append(ChecklistItem(
                task=task, category="verification", priority=priority
            ))
        
        # Get attack-specific checklist
        attack_checklist = self.ATTACK_CHECKLISTS.get(attack_type, self.DEFAULT_CHECKLIST)
        
        for category, items in attack_checklist.items():
            for task, priority in items:
                # Adjust priority based on severity
                if severity == "critical":
                    priority = max(1, priority - 1)
                elif severity == "low":
                    priority = min(5, priority + 1)
                
                checklist.items.append(ChecklistItem(
                    task=task, category=category, priority=priority
                ))
        
        # Add high-confidence specific items
        if confidence >= 0.95:
            checklist.items.insert(0, ChecklistItem(
                task="HIGH CONFIDENCE: Prioritize immediate response",
                category="response", priority=1
            ))
        
        return checklist
    
    def generate_fp_checklist(self, alert_id: str) -> AnalystChecklist:
        """Generate checklist for potential false positive review."""
        checklist = AnalystChecklist(
            alert_id=alert_id,
            attack_type="potential_false_positive",
            severity="low"
        )
        
        fp_items = [
            ("Check if source is internal/trusted", "verification", 1),
            ("Review business context of request", "verification", 1),
            ("Check if pattern matches known benign activity", "verification", 2),
            ("Verify user role and permissions", "investigation", 2),
            ("Check historical activity from source", "investigation", 2),
            ("Mark as false positive if confirmed", "response", 3),
            ("Add to whitelist if recurring FP", "response", 3),
            ("Document FP pattern for model improvement", "documentation", 4),
        ]
        
        for task, category, priority in fp_items:
            checklist.items.append(ChecklistItem(
                task=task, category=category, priority=priority
            ))
        
        return checklist


def generate_checklist(alert_id: str, attack_type: str, severity: str,
                      confidence: float = 0.0) -> Dict:
    """Convenience function to generate checklist."""
    generator = ChecklistGenerator()
    checklist = generator.generate(alert_id, attack_type, severity, confidence)
    return checklist.to_dict()
