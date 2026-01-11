"""Generate targeted benign data to fix high FP categories."""
import json
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent / 'datasets' / 'benign_60m'

# Templates for high-FP categories
SUBJECTS = ['The manager', 'A customer', 'Our team', 'The system', 'This product', 
            'The service', 'Your order', 'The report', 'Our company', 'The user',
            'Your request', 'The application', 'Our support', 'The department']

VERBS = ['has completed', 'needs to update', 'successfully handled', 'is waiting for',
         'has been approved', 'requires', 'is processing', 'will review', 'confirmed',
         'received', 'sent', 'delivered', 'scheduled', 'finalized', 'verified']

OBJECTS = ['your account', 'this transaction', 'the document', 'your submission',
           'the application', 'the data', 'the request', 'your account', 'the payment',
           'your profile', 'the invoice', 'your subscription', 'the contract']

FIRST_NAMES = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 
               'Linda', 'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan',
               'Joseph', 'Jessica', 'Thomas', 'Sarah', 'Charles', 'Karen', 'Christopher',
               'Lisa', 'Daniel', 'Nancy', 'Matthew', 'Betty', 'Anthony', 'Margaret',
               'Mark', 'Sandra', 'Donald', 'Ashley', 'Steven', 'Kimberly', 'Paul', 'Emily']

LAST_NAMES = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
              'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
              'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
              'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez']

STREET_TYPES = ['Street', 'Avenue', 'Boulevard', 'Drive', 'Lane', 'Road', 'Way', 'Court', 'Place']
STREET_NAMES = ['Main', 'Oak', 'Maple', 'Cedar', 'Pine', 'Elm', 'Washington', 'Park', 
                'Lake', 'Hill', 'Forest', 'River', 'Spring', 'Valley', 'Sunset', 'Highland']
CITIES = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
          'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
          'Fort Worth', 'Columbus', 'Charlotte', 'Seattle', 'Denver', 'Boston', 'Portland']
STATES = ['NY', 'CA', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA', 'AZ', 'MA', 'CO']

DOMAINS = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'company.com',
           'business.org', 'work.net', 'mail.com', 'email.com', 'corp.io']

PRODUCT_TYPES = ['Wireless', 'Premium', 'Professional', 'Ultra', 'Smart', 'Digital', 
                 'Portable', 'Advanced', 'Classic', 'Modern', 'Compact', 'Heavy-Duty']
PRODUCT_ITEMS = ['Headphones', 'Keyboard', 'Mouse', 'Monitor', 'Speaker', 'Camera',
                 'Laptop Stand', 'USB Hub', 'Charger', 'Cable', 'Adapter', 'Case',
                 'Backpack', 'Desk Lamp', 'Chair', 'Desk', 'Microphone', 'Webcam']
PRODUCT_FEATURES = ['with Bluetooth', 'with USB-C', 'with LED', 'with Noise Cancellation',
                    'with Fast Charging', 'with Ergonomic Design', 'with RGB Lighting',
                    'with Wireless Connectivity', 'with Touch Controls', 'with Voice Control']


def gen_sentence():
    """Generate business sentence."""
    subj = random.choice(SUBJECTS)
    verb = random.choice(VERBS)
    obj = random.choice(OBJECTS)
    ref = f"#{random.randint(100000, 999999)}" if random.random() > 0.5 else ""
    return f"{subj} {verb} {obj} {ref}".strip()


def gen_name():
    """Generate full name."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    if random.random() > 0.7:
        middle = random.choice(FIRST_NAMES)[0]
        return f"{first} {middle}. {last}"
    return f"{first} {last}"


def gen_address():
    """Generate physical address."""
    num = random.randint(1, 9999)
    street = random.choice(STREET_NAMES)
    st_type = random.choice(STREET_TYPES)
    city = random.choice(CITIES)
    state = random.choice(STATES)
    zip_code = random.randint(10000, 99999)
    
    if random.random() > 0.5:
        apt = f"Apt {random.randint(1, 999)}"
        return f"{num} {street} {st_type}, {apt}, {city}, {state} {zip_code}"
    return f"{num} {street} {st_type}, {city}, {state} {zip_code}"


def gen_email():
    """Generate email address."""
    first = random.choice(FIRST_NAMES).lower()
    last = random.choice(LAST_NAMES).lower()
    domain = random.choice(DOMAINS)
    
    patterns = [
        f"{first}.{last}@{domain}",
        f"{first}{last}@{domain}",
        f"{first}_{last}@{domain}",
        f"{first[0]}{last}@{domain}",
        f"{first}{random.randint(1, 99)}@{domain}",
    ]
    return random.choice(patterns)


def gen_product():
    """Generate product description."""
    ptype = random.choice(PRODUCT_TYPES)
    item = random.choice(PRODUCT_ITEMS)
    feature = random.choice(PRODUCT_FEATURES) if random.random() > 0.4 else ""
    price = f"${random.randint(10, 500)}.{random.randint(0, 99):02d}"
    
    templates = [
        f"{ptype} {item} {feature} - {price}",
        f"{item} ({ptype}) {feature}",
        f"{ptype} {item}: {feature}",
        f"Buy {ptype} {item} {feature} for only {price}",
        f"{item} - {ptype} Edition {feature}",
    ]
    return random.choice(templates).strip()


def generate_fp_fix_data(samples_per_category=2_000_000):
    """Generate benign data for high-FP categories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    generators = {
        'sentences': gen_sentence,
        'names': gen_name,
        'addresses': gen_address,
        'emails': gen_email,
        'products': gen_product,
    }
    
    for category, gen_func in generators.items():
        output_file = OUTPUT_DIR / f'{category}_2m.jsonl'
        print(f"\nGenerating {category}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for _ in tqdm(range(samples_per_category), desc=category):
                text = gen_func()
                f.write(json.dumps({'text': text, 'label': 0}) + '\n')
        
        size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"  Saved: {output_file.name} ({size_mb:.1f} MB)")
    
    print(f"\n✓ Generated {len(generators) * samples_per_category:,} total samples")


if __name__ == '__main__':
    generate_fp_fix_data()
