"""Generate curated benign data for payload classifier training."""
import random
import string
import json
from pathlib import Path

# Common first and last names
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Emma", "Oliver", "Ava", "Liam", "Sophia",
    "Noah", "Isabella", "Ethan", "Mia", "Lucas", "Charlotte", "Mason", "Amelia",
    "Carlos", "Maria", "Ahmed", "Fatima", "Wei", "Yuki", "Raj", "Priya", "Ivan", "Anna"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores"
]

# Common domains for emails
EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
    "mail.com", "protonmail.com", "aol.com", "live.com", "msn.com",
    "company.com", "work.org", "university.edu", "business.net"
]

# Common sentences and phrases
SENTENCES = [
    "Hello, how are you today?",
    "Thank you for your help.",
    "Please let me know if you have any questions.",
    "I would like to schedule a meeting.",
    "The weather is nice today.",
    "Can you send me the report?",
    "Looking forward to hearing from you.",
    "Best regards",
    "Have a great day!",
    "Please find attached the document.",
    "I agree with your proposal.",
    "Let me check and get back to you.",
    "That sounds like a good idea.",
    "Could you please clarify?",
    "I'll be there in 10 minutes.",
    "Thanks for the update.",
    "See you tomorrow.",
    "Happy birthday!",
    "Congratulations on your achievement.",
    "Welcome to our team.",
    "Please review and approve.",
    "I need more information.",
    "The project is on track.",
    "Great work everyone!",
    "Let's discuss this further.",
    "I appreciate your feedback.",
    "Sorry for the delay.",
    "Here's the summary.",
    "Please confirm receipt.",
    "Looking forward to our collaboration.",
]

# Product names and descriptions
PRODUCTS = [
    "iPhone 15 Pro Max 256GB",
    "Samsung Galaxy S24 Ultra",
    "MacBook Pro 14-inch M3",
    "Sony WH-1000XM5 Headphones",
    "Nike Air Jordan 1 Retro",
    "Adidas Ultraboost Running Shoes",
    "PlayStation 5 Digital Edition",
    "Nintendo Switch OLED",
    "Canon EOS R5 Camera",
    "Dyson V15 Vacuum Cleaner",
    "KitchenAid Stand Mixer",
    "Instant Pot Duo 7-in-1",
    "Kindle Paperwhite 2024",
    "Apple Watch Series 9",
    "Bose QuietComfort Earbuds",
]

# Search queries
SEARCH_QUERIES = [
    "best restaurants near me",
    "weather forecast tomorrow",
    "how to learn python",
    "cheap flights to new york",
    "recipe for chocolate cake",
    "movie showtimes",
    "latest news today",
    "online shopping deals",
    "fitness tips for beginners",
    "home renovation ideas",
    "best laptop 2024",
    "healthy breakfast recipes",
    "job openings remote",
    "travel destinations europe",
    "how to fix wifi connection",
    "birthday gift ideas",
    "stock market today",
    "new movies streaming",
    "diy home projects",
    "best coffee shops",
]

# Street names
STREETS = ["Main", "Oak", "Maple", "Cedar", "Pine", "Elm", "Washington", "Lake", "Hill", "Park"]
STREET_TYPES = ["St", "Ave", "Blvd", "Dr", "Ln", "Rd", "Way", "Ct", "Pl"]
CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "San Diego", "Dallas", "Austin", "Seattle", "Denver"]
STATES = ["NY", "CA", "IL", "TX", "AZ", "FL", "WA", "CO", "GA", "NC"]


def generate_name():
    """Generate a realistic name."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    formats = [
        f"{first} {last}",
        f"{first}",
        f"{last}",
        f"{first} {random.choice(string.ascii_uppercase)}. {last}",
        f"{first.lower()}_{last.lower()}",
    ]
    return random.choice(formats)


def generate_email():
    """Generate a realistic email address."""
    first = random.choice(FIRST_NAMES).lower()
    last = random.choice(LAST_NAMES).lower()
    domain = random.choice(EMAIL_DOMAINS)
    num = random.randint(1, 999) if random.random() < 0.3 else ""
    
    formats = [
        f"{first}.{last}{num}@{domain}",
        f"{first}{last}{num}@{domain}",
        f"{first[0]}{last}{num}@{domain}",
        f"{first}_{last}{num}@{domain}",
        f"{first}{num}@{domain}",
    ]
    return random.choice(formats)


def generate_phone():
    """Generate a realistic phone number."""
    area = random.randint(200, 999)
    prefix = random.randint(200, 999)
    line = random.randint(1000, 9999)
    
    formats = [
        f"({area}) {prefix}-{line}",
        f"{area}-{prefix}-{line}",
        f"+1-{area}-{prefix}-{line}",
        f"{area}.{prefix}.{line}",
        f"+1 ({area}) {prefix}-{line}",
    ]
    return random.choice(formats)


def generate_address():
    """Generate a realistic address."""
    num = random.randint(1, 9999)
    street = random.choice(STREETS)
    st_type = random.choice(STREET_TYPES)
    city = random.choice(CITIES)
    state = random.choice(STATES)
    zip_code = random.randint(10000, 99999)
    
    formats = [
        f"{num} {street} {st_type}",
        f"{num} {street} {st_type}, {city}, {state}",
        f"{num} {street} {st_type}, {city}, {state} {zip_code}",
        f"{city}, {state}",
        f"{city}, {state} {zip_code}",
    ]
    return random.choice(formats)


def generate_date():
    """Generate a date in various formats."""
    year = random.randint(2020, 2025)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    
    formats = [
        f"{year}-{month:02d}-{day:02d}",
        f"{month:02d}/{day:02d}/{year}",
        f"{day:02d}/{month:02d}/{year}",
        f"{months[month-1]} {day}, {year}",
        f"{day} {months[month-1]} {year}",
        f"{month}/{day}/{year}",
    ]
    return random.choice(formats)


def generate_username():
    """Generate a realistic username."""
    first = random.choice(FIRST_NAMES).lower()
    num = random.randint(1, 9999) if random.random() < 0.5 else ""
    
    formats = [
        f"{first}{num}",
        f"{first}_{num}" if num else first,
        f"user{random.randint(1000, 9999)}",
        f"{first}{random.choice(['_x', '_official', '2024', '_real'])}",
    ]
    return random.choice(formats)


def generate_comment():
    """Generate a realistic user comment."""
    comments = [
        "Great product, highly recommend!",
        "Fast shipping, exactly as described.",
        "Good quality for the price.",
        "Would buy again.",
        "Not what I expected, but okay.",
        "Excellent customer service!",
        "Works perfectly.",
        "Very satisfied with my purchase.",
        "Could be better, but it's fine.",
        "Amazing! Five stars!",
        "Decent product overall.",
        "Love it! Thanks!",
        "Quick delivery, thanks!",
        "As advertised.",
        "Happy with this purchase.",
    ]
    return random.choice(comments)


def generate_json_snippet():
    """Generate a benign JSON snippet."""
    templates = [
        {"status": "ok", "message": "Success"},
        {"id": random.randint(1, 1000), "name": generate_name()},
        {"count": random.randint(1, 100), "page": random.randint(1, 10)},
        {"result": True, "data": []},
        {"user": generate_username(), "active": True},
        {"items": [], "total": random.randint(0, 50)},
        {"error": False, "code": 200},
    ]
    return json.dumps(random.choice(templates))


def generate_benign_dataset(output_dir, samples_per_category=5000):
    """Generate complete benign dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    # Generate each category
    categories = {
        'sentences': lambda: random.choice(SENTENCES),
        'names': generate_name,
        'emails': generate_email,
        'phones': generate_phone,
        'addresses': generate_address,
        'dates': generate_date,
        'usernames': generate_username,
        'products': lambda: random.choice(PRODUCTS),
        'search_queries': lambda: random.choice(SEARCH_QUERIES),
        'comments': generate_comment,
        'json': generate_json_snippet,
    }
    
    for category, generator in categories.items():
        samples = []
        for _ in range(samples_per_category):
            samples.append(generator())
        
        # Save category file
        with open(output_dir / f"{category}.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(samples))
        
        all_samples.extend(samples)
        print(f"Generated {len(samples)} {category}")
    
    # Save combined file
    random.shuffle(all_samples)
    with open(output_dir / "all_benign.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_samples))
    
    print(f"\nTotal: {len(all_samples)} benign samples")
    print(f"Saved to: {output_dir}")
    
    return all_samples


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    output = base / "datasets" / "curated_benign"
    generate_benign_dataset(output, samples_per_category=5000)
