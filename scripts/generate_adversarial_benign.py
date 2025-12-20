"""Generate adversarial benign data - edge cases that look malicious but are benign."""
import random
from pathlib import Path

# Names with apostrophes (triggers SQL injection false positives)
IRISH_NAMES = ["O'Brien", "O'Connor", "O'Neill", "O'Sullivan", "O'Reilly", "O'Donnell", "O'Malley"]
FRENCH_NAMES = ["D'Angelo", "D'Arcy", "D'Costa", "L'Amour", "D'Souza"]
OTHER_APOSTROPHE = ["McDonald's", "Denny's", "Macy's", "Lowe's", "Kohl's"]

# SQL keywords in benign context
SQL_BENIGN_TEMPLATES = [
    "SELECT your favorite dish from our menu",
    "Please SELECT the items you want",
    "FROM our collection of products",
    "WHERE can I find the bathroom?",
    "DROP by our store anytime",
    "DELETE old emails from inbox",
    "UPDATE your profile settings",
    "INSERT your name here",
    "JOIN us for dinner tonight",
    "ORDER BY phone or online",
    "UNION of workers meeting",
    "SELECT ALL that apply",
    "FROM 9am to 5pm",
    "WHERE are you going?",
]

# Security discussions (benign content about security)
SECURITY_DISCUSSIONS = [
    "How to prevent SQL injection attacks",
    "XSS vulnerability explained for beginners",
    "Best practices for command injection prevention",
    "Understanding cross-site scripting",
    "Security audit report summary",
    "Penetration testing methodology",
    "OWASP Top 10 vulnerabilities",
    "Secure coding guidelines",
    "Input validation best practices",
    "Sanitizing user input tutorial",
]

# Code snippets (operators look like injection)
CODE_SNIPPETS = [
    "if (x || y) return true",
    "while (a && b) continue",
    "result = x == y ? 1 : 0",
    "count = count + 1",
    "price * quantity = total",
    "if (user != null) process()",
    "for (i = 0; i < 10; i++)",
    "return a || b || c",
    "flag = (x > 0) && (y < 10)",
    "value = input ?? default",
]

# File paths (backslashes trigger command injection)
FILE_PATHS = [
    "C:\\Users\\Admin\\Documents",
    "C:\\Program Files\\Application",
    "D:\\Projects\\MyApp\\src",
    "C:\\Windows\\System32",
    "/home/user/documents",
    "/var/log/application.log",
    "/etc/nginx/nginx.conf",
    "~/Downloads/file.pdf",
    "./scripts/run.sh",
    "../config/settings.json",
]

# Math expressions
MATH_EXPRESSIONS = [
    "1+1=2", "2*3=6", "10/2=5", "5-3=2",
    "x+y=z", "a*b=c", "price*qty", "total/count",
    "100%", "50% off", "2x + 3y = 10",
    "E=mc^2", "a^2 + b^2 = c^2",
]

# Pricing with OR (triggers SQL injection)
PRICING_TEXT = [
    "$100 OR best offer",
    "$50 or nearest offer",
    "Buy 2 OR more get discount",
    "Call OR text for info",
    "Cash OR credit accepted",
    "Pickup OR delivery available",
    "Small OR medium size",
    "Red OR blue color",
    "Option A OR option B",
    "Yes OR no answer required",
]

# Script mentions (triggers XSS detection)
SCRIPT_MENTIONS = [
    "The script was amazing",
    "Movie script writer",
    "Follow the script carefully",
    "Script for the play",
    "Bash script tutorial",
    "Python script example",
    "Read the script aloud",
    "Original script by author",
    "Script supervisor on set",
    "Unscripted moment",
]

# HTML in educational context
HTML_EDUCATIONAL = [
    "Use the <div> tag for containers",
    "The <p> element for paragraphs",
    "Add <img> for images",
    "Create links with <a> tag",
    "The <body> contains content",
    "Use <head> for metadata",
    "Tables use <tr> and <td>",
    "Lists: <ul> and <ol>",
    "Forms need <input> fields",
    "The <span> is inline",
]

# Command tutorials
COMMAND_TUTORIALS = [
    "The ls command lists files",
    "Use cat to view file contents",
    "The cd command changes directory",
    "Run pwd to see current path",
    "Use grep to search text",
    "The rm command removes files",
    "Copy files with cp command",
    "Move files using mv",
    "Create directories with mkdir",
    "The echo command prints text",
]


def generate_names_with_apostrophes(n=2000):
    """Generate names containing apostrophes."""
    samples = []
    all_names = IRISH_NAMES + FRENCH_NAMES + OTHER_APOSTROPHE
    
    for _ in range(n):
        name = random.choice(all_names)
        templates = [
            name,
            f"Hello {name}",
            f"Meeting with {name}",
            f"{name}'s order",
            f"Contact: {name}",
            f"Dear {name},",
        ]
        samples.append(random.choice(templates))
    return samples


def generate_sql_benign(n=2000):
    """Generate benign text containing SQL keywords."""
    samples = []
    for _ in range(n):
        samples.append(random.choice(SQL_BENIGN_TEMPLATES))
    return samples


def generate_security_content(n=2000):
    """Generate benign security educational content."""
    samples = []
    for _ in range(n):
        base = random.choice(SECURITY_DISCUSSIONS)
        templates = [base, f"Article: {base}", f"Learn about {base.lower()}"]
        samples.append(random.choice(templates))
    return samples


def generate_code_snippets(n=2000):
    """Generate benign code snippets."""
    samples = []
    for _ in range(n):
        samples.append(random.choice(CODE_SNIPPETS))
    return samples


def generate_file_paths(n=2000):
    """Generate benign file paths."""
    samples = []
    for _ in range(n):
        samples.append(random.choice(FILE_PATHS))
    return samples


def generate_math(n=2000):
    """Generate math expressions."""
    samples = []
    for _ in range(n):
        samples.append(random.choice(MATH_EXPRESSIONS))
    return samples


def generate_pricing(n=2000):
    """Generate pricing text with OR."""
    samples = []
    for _ in range(n):
        samples.append(random.choice(PRICING_TEXT))
    return samples


def generate_script_mentions(n=2000):
    """Generate benign mentions of 'script'."""
    samples = []
    for _ in range(n):
        samples.append(random.choice(SCRIPT_MENTIONS))
    return samples


def generate_html_educational(n=2000):
    """Generate HTML educational content."""
    samples = []
    for _ in range(n):
        samples.append(random.choice(HTML_EDUCATIONAL))
    return samples


def generate_command_tutorials(n=2000):
    """Generate command line tutorials."""
    samples = []
    for _ in range(n):
        samples.append(random.choice(COMMAND_TUTORIALS))
    return samples


def main():
    """Generate all adversarial benign data."""
    base_path = Path(__file__).parent.parent
    output_dir = base_path / "datasets" / "curated_benign" / "adversarial"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(" Adversarial Benign Data Generation")
    print("=" * 60)
    
    generators = {
        "names_apostrophe": generate_names_with_apostrophes,
        "sql_benign": generate_sql_benign,
        "security_content": generate_security_content,
        "code_snippets": generate_code_snippets,
        "file_paths": generate_file_paths,
        "math_expressions": generate_math,
        "pricing_or": generate_pricing,
        "script_mentions": generate_script_mentions,
        "html_educational": generate_html_educational,
        "command_tutorials": generate_command_tutorials,
    }
    
    all_samples = []
    
    for name, generator in generators.items():
        samples = generator(2000)
        
        # Save individual file
        with open(output_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(samples))
        
        all_samples.extend(samples)
        print(f"  ✓ Generated {len(samples)} {name}")
    
    # Save combined file
    random.shuffle(all_samples)
    with open(output_dir / "all_adversarial.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_samples))
    
    print(f"\nTotal: {len(all_samples)} adversarial benign samples")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
