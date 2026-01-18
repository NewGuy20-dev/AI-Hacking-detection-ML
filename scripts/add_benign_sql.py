#!/usr/bin/env python3
"""Add benign SQL samples to payload training data."""
import json
import random

BENIGN_SQL = [
    "SELECT * FROM products WHERE price < 100",
    "SELECT name, email FROM users WHERE id = 123",
    "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
    "SELECT title, author FROM books WHERE year > 2020",
    "UPDATE cart SET quantity = 2 WHERE user_id = 456",
    "UPDATE profile SET last_login = NOW() WHERE id = 789",
    "UPDATE inventory SET stock = stock - 1 WHERE product_id = 101",
    "INSERT INTO orders (user, total) VALUES ('john', 99.99)",
    "INSERT INTO logs (timestamp, message) VALUES (NOW(), 'User logged in')",
    "INSERT INTO comments (post_id, text) VALUES (42, 'Great article!')",
    "DELETE FROM wishlist WHERE item_id = 789",
    "DELETE FROM sessions WHERE expires_at < NOW()",
    "DELETE FROM temp_data WHERE created_at < DATE_SUB(NOW(), INTERVAL 1 DAY)",
    "SELECT p.name, c.name FROM products p JOIN categories c ON p.category_id = c.id",
    "SELECT u.username, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id",
]

def main():
    input_file = "datasets/benign_60m/payload_benign_expansion.jsonl"
    output_file = "datasets/benign_60m/payload_benign_expansion_with_sql.jsonl"
    
    # Calculate how many SQL samples to add (5% of total)
    total_lines = 20_800_000
    sql_samples = int(total_lines * 0.05)
    
    print(f"Adding {sql_samples:,} benign SQL samples to {total_lines:,} existing samples...")
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        # Copy existing data
        for i, line in enumerate(fin):
            fout.write(line)
            if (i + 1) % 1_000_000 == 0:
                print(f"  Copied {i+1:,} lines...")
        
        # Add SQL samples
        for i in range(sql_samples):
            sql = random.choice(BENIGN_SQL)
            entry = {"text": sql, "label": 0}
            fout.write(json.dumps(entry) + '\n')
            if (i + 1) % 100_000 == 0:
                print(f"  Added {i+1:,} SQL samples...")
    
    print(f"\n✓ Created {output_file}")
    print(f"  Total samples: {total_lines + sql_samples:,}")
    print(f"  SQL samples: {sql_samples:,} (~5%)")
    print(f"\nTo use: Replace the old file or update training script to use new file")

if __name__ == "__main__":
    main()
