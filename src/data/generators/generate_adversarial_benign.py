"""Generate adversarial benign data - edge cases that look malicious but are benign."""
import random
from pathlib import Path

# Names with apostrophes (triggers SQL injection false positives)
IRISH_NAMES = ["O'Brien", "O'Connor", "O'Neill", "O'Sullivan", "O'Reilly", "O'Donnell", "O'Malley"]
FRENCH_NAMES = ["D'Angelo", "D'Arcy", "D'Costa", "L'Amour", "D'Souza"]
OTHER_APOSTROPHE = ["McDonald's", "Denny's", "Macy's", "Lowe's", "Kohl's"]
CONTRACTIONS = ["It's great", "It's amazing", "That's wonderful", "Don't worry", "Can't wait", "Won't stop", "Isn't it", "Aren't you", "Wasn't that", "Weren't they"]

# SQL keywords in benign context
SQL_BENIGN_TEMPLATES = [
    "SELECT your favorite dish from our menu",
    "Please SELECT the items you want",
    "FROM our collection of products",
    "WHERE can I find the bathroom?",
    "DROP by our store anytime",
    "DROP by anytime",
    "DROP in for a visit",
    "DROP me a line",
    "DELETE old emails from inbox",
    "UPDATE your profile settings",
    "INSERT your name here",
    "JOIN us for dinner tonight",
    "ORDER BY phone or online",
    "UNION of workers meeting",
    "SELECT ALL that apply",
    "FROM 9am to 5pm",
    "WHERE are you going?",
    "SELECT * FROM menu - our specials today",
    "SELECT from our wine list",
    "Please SELECT your preferred date",
    "FROM the bottom of my heart",
    "WHERE do you want to eat?",
    "SELECT one or more options below",
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
    "~/Desktop/file.txt",
    "~/Documents/report.doc",
    "./scripts/run.sh",
    "../config/settings.json",
]

# Math expressions
MATH_EXPRESSIONS = [
    "1+1=2", "2*3=6", "10/2=5", "5-3=2",
    "x+y=z", "a*b=c", "price*qty", "total/count",
    "100%", "50% off", "2x + 3y = 10",
    "E=mc^2", "a^2 + b^2 = c^2",
    "x = y + z", "a = b + c", "n = m + 1",
    "result = x + y", "sum = a + b",
    "x = y - z", "x = y * z", "x = y / z",
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
    "Option A OR B",
    "Yes OR no answer required",
    "$200 or best offer - must sell!",
    "Price negotiable OR trade accepted",
    "Available for pickup OR shipping",
    "Contact via email OR phone",
    "Morning OR afternoon delivery",
    "This OR that",
    "One OR the other",
    "Now OR later",
    "Here OR there",
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
    "<img> adds images",
    "<img> tag for pictures",
    "Create links with <a> tag",
    "The <body> contains content",
    "Use <head> for metadata",
    "Tables use <tr> and <td>",
    "Lists: <ul> and <ol>",
    "Forms need <input> fields",
    "The <span> is inline",
    "<script> for JavaScript",
    "<style> for CSS",
    "<link> for stylesheets",
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

# === NEW: Targeted edge case generators ===

# Emails with dots (triggers false positives)
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "company.com", "work.org", "university.edu", "support.io", "domain.net", "mail.co"]
FIRST_NAMES = ["john", "jane", "mike", "sarah", "david", "emma", "chris", "lisa", "tom", "anna", "contact", "info", "support", "admin", "help"]
LAST_NAMES = ["smith", "doe", "johnson", "williams", "brown", "jones", "davis", "miller", "wilson", "moore", "us", "team", "sales", "service"]

# Heart emoji and emoticons (< triggers XSS detection)
HEART_MESSAGES = [
    "<3 love this product",
    "<3 amazing service",
    "I <3 this so much",
    "<3 <3 <3 best ever",
    "Love it <3",
    "<3 thank you!",
    "So cute <3",
    "<3 highly recommend",
    "Perfect <3 <3",
    "<3 five stars",
    "Absolutely <3 it",
    "<3 will buy again",
    "My favorite <3",
    "<3 great quality",
    "Love love love <3",
    "<33 obsessed",
    "So happy <3",
    "<3 beautiful",
    "Amazing <3 <3",
    "Best purchase <3",
]

# Restaurant/menu SQL-like text
MENU_SQL_LIKE = [
    "SELECT from our appetizer menu",
    "SELECT * FROM menu - today's specials",
    "Please SELECT your entree",
    "SELECT any two sides",
    "SELECT from breakfast or lunch menu",
    "Our SELECT menu features local ingredients",
    "SELECT your protein: chicken, beef, or fish",
    "SELECT from our wine list",
    "Daily SELECT: chef's choice",
    "SELECT combo meal",
    "SELECT size: small, medium, large",
    "SELECT your toppings",
    "SELECT from gluten-free options",
    "SELECT dessert from the cart",
    "SELECT your cooking preference",
]

# Code snippets (common patterns that trigger FPs)
CODE_BENIGN = [
    "function test() {}",
    "function getData() { return data; }",
    "const result = fetch(url);",
    "for i in range(10):",
    "for (let i = 0; i < 10; i++)",
    "if (x > 0) return true;",
    "while (running) { process(); }",
    "def main():",
    "class MyClass:",
    "import numpy as np",
    "console.log('hello');",
    "print('Hello World')",
    "return response.json()",
    "async function load() {}",
    "export default App;",
    "module.exports = router;",
    "const express = require('express');",
    "app.listen(3000);",
    "router.get('/', handler);",
    "useState(false)",
]

# URLs (benign patterns)
URL_BENIGN = [
    "https://www.google.com",
    "https://github.com/user/repo",
    "https://stackoverflow.com/questions",
    "https://www.amazon.com/product",
    "https://en.wikipedia.org/wiki/Topic",
    "https://www.youtube.com/watch",
    "https://twitter.com/user",
    "https://www.linkedin.com/in/profile",
    "https://docs.python.org/3/",
    "https://reactjs.org/docs",
    "https://api.example.com/v1/users",
    "https://cdn.jsdelivr.net/npm/",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
    "https://www.reddit.com/r/programming",
]

# Product reviews (common phrases)
REVIEWS_BENIGN = [
    "Exactly as described",
    "Five stars!",
    "Great product!",
    "Fast shipping",
    "Would recommend",
    "Perfect fit",
    "Excellent quality",
    "Best purchase ever",
    "Works great",
    "Very satisfied",
    "Exceeded expectations",
    "Good value",
    "As advertised",
    "Quick delivery",
    "Love this item",
    "Have a wonderful day",
    "Have a great day",
    "Have a nice day",
    "Best wishes",
    "Thank you so much",
]

# Unix paths
UNIX_PATHS = [
    "/home/user/downloads",
    "/var/log/app.log",
    "/etc/nginx/nginx.conf",
    "/usr/local/bin/python",
    "/opt/app/config.json",
    "/tmp/cache/data",
    "/srv/www/html",
    "/root/.bashrc",
    "/home/admin/scripts",
    "/var/www/html/index.html",
]


def generate_names_with_apostrophes(n=2000):
    """Generate names containing apostrophes."""
    samples = []
    all_names = IRISH_NAMES + FRENCH_NAMES + OTHER_APOSTROPHE + CONTRACTIONS
    
    for _ in range(n):
        name = random.choice(all_names)
        templates = [
            name,
            f"Hello {name}",
            f"Meeting with {name}",
            f"{name}'s order" if name not in CONTRACTIONS else name,
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


def generate_emails_with_dots(n=5000):
    """Generate email addresses with dots (triggers FP)."""
    samples = []
    for _ in range(n):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        domain = random.choice(EMAIL_DOMAINS)
        num = random.randint(1, 999) if random.random() < 0.3 else ""
        mid = random.choice(LAST_NAMES) if random.random() < 0.3 else ""
        
        # Various email formats with dots
        if mid:
            email = f"{first}.{mid}.{last}{num}@{domain}"
        else:
            email = f"{first}.{last}{num}@{domain}"
        
        formats = [
            email,
            f"Contact: {email}",
            f"Email me at {email}",
            f"Send to {email}",
            f"Reply to: {email}",
            f"{email} is my email",
        ]
        samples.append(random.choice(formats))
    return samples


def generate_heart_emoji(n=5000):
    """Generate messages with <3 heart emoji."""
    samples = []
    for _ in range(n):
        samples.append(random.choice(HEART_MESSAGES))
    return samples


def generate_menu_sql(n=5000):
    """Generate restaurant menu SQL-like text."""
    samples = []
    for _ in range(n):
        samples.append(random.choice(MENU_SQL_LIKE))
    return samples


def generate_code_benign(n=5000):
    """Generate benign code snippets."""
    samples = []
    for _ in range(n):
        samples.append(random.choice(CODE_BENIGN))
    return samples


def generate_url_benign(n=5000):
    """Generate benign URLs."""
    samples = []
    for _ in range(n):
        base = random.choice(URL_BENIGN)
        # Add some variation
        if random.random() < 0.3:
            base += f"/{random.randint(1, 9999)}"
        samples.append(base)
    return samples


def generate_reviews_benign(n=5000):
    """Generate benign product reviews."""
    samples = []
    for _ in range(n):
        samples.append(random.choice(REVIEWS_BENIGN))
    return samples


def generate_unix_paths(n=5000):
    """Generate benign unix paths."""
    samples = []
    for _ in range(n):
        samples.append(random.choice(UNIX_PATHS))
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
        # Targeted generators for edge cases
        "emails_with_dots": generate_emails_with_dots,
        "heart_emoji": generate_heart_emoji,
        "menu_sql_like": generate_menu_sql,
        "code_benign": generate_code_benign,
        "url_benign": generate_url_benign,
        "reviews_benign": generate_reviews_benign,
        "unix_paths": generate_unix_paths,
    }
    
    all_samples = []
    
    for name, generator in generators.items():
        # Use larger sample size for targeted generators
        if name in ["emails_with_dots", "heart_emoji", "menu_sql_like", "code_benign", "url_benign", "reviews_benign", "unix_paths"]:
            n = 5000
        else:
            n = 2000
        samples = generator(n)
        
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
