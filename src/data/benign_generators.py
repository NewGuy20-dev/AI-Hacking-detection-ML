"""Benign data generators for expanding training dataset."""
import random
import string
from typing import List, Iterator
from datetime import datetime, timedelta


# Common data pools
TABLES = ["users", "products", "orders", "customers", "inventory", "sessions", 
          "payments", "reviews", "categories", "logs", "employees", "tasks"]
COLUMNS = ["id", "name", "email", "created_at", "status", "price", "quantity", 
           "description", "user_id", "title", "content", "is_active"]
PATHS = ["/home", "/var/log", "/tmp", "/opt", "/etc", "/usr/local", ".", "./src", "./data"]
EXTENSIONS = [".py", ".js", ".json", ".yaml", ".txt", ".log", ".csv", ".xml"]
METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]
RESOURCES = ["users", "products", "orders", "posts", "comments", "files", "settings"]
LEVELS = ["DEBUG", "INFO", "WARN", "ERROR"]
SERVICES = ["nginx", "apache", "mysql", "redis", "app", "api", "worker"]


def generate_urls(domains: List[str], count: int) -> Iterator[str]:
    """Generate URL variations from domain list."""
    paths = ["", "/", "/about", "/contact", "/products", "/services", "/blog",
             "/search", "/user/profile", "/api/v1/data", "/login", "/register",
             "/faq", "/help", "/terms", "/privacy", "/sitemap.xml", "/robots.txt",
             "/pricing", "/features", "/docs", "/support", "/careers", "/news",
             "/events", "/partners", "/investors", "/press", "/legal", "/security"]
    subdomains = ["", "www.", "api.", "cdn.", "static.", "m.", "app.", "dev.", 
                  "staging.", "beta.", "admin.", "dashboard.", "mail.", "shop."]
    params = ["", "?q=test", "?page=1", "?id=123", "?sort=asc", "?limit=10",
              "?ref=home", "?utm_source=google", "?lang=en", "?v=2"]
    
    for i in range(count):
        domain = domains[i % len(domains)]
        sub = subdomains[i % len(subdomains)]
        path = paths[i % len(paths)]
        param = params[i % len(params)]
        uid = f"{i:08x}"
        
        # Add unique suffix to ensure no duplicates
        if random.random() > 0.5:
            yield f"https://{sub}{domain}{path}{param}&sid={uid}"
        else:
            yield f"https://{sub}{domain}{path}/{uid}{param}"


def generate_sql(count: int) -> Iterator[str]:
    """Generate legitimate SQL queries."""
    templates = [
        "SELECT {col} FROM {table}",
        "SELECT {col}, {col2} FROM {table} WHERE {col} = '{val}'",
        "SELECT * FROM {table} ORDER BY {col} LIMIT {num}",
        "SELECT COUNT(*) FROM {table}",
        "SELECT {col} FROM {table} WHERE {col} IN ({vals})",
        "INSERT INTO {table} ({col}, {col2}) VALUES ('{val}', '{val2}')",
        "UPDATE {table} SET {col} = '{val}' WHERE id = {num}",
        "DELETE FROM {table} WHERE {col} = '{val}'",
        "SELECT t1.{col} FROM {table} t1 JOIN {table2} t2 ON t1.id = t2.{col}_id",
        "SELECT {col}, COUNT(*) FROM {table} GROUP BY {col}",
    ]
    
    for i in range(count):
        uid = f"{i:08x}"
        t = templates[i % len(templates)]
        col = COLUMNS[i % len(COLUMNS)]
        col2 = COLUMNS[(i + 3) % len(COLUMNS)]
        table = TABLES[i % len(TABLES)]
        table2 = TABLES[(i + 2) % len(TABLES)]
        num = (i % 10000) + 1
        
        yield t.format(
            col=col, col2=col2,
            table=table, table2=table2,
            val=uid,
            val2=f"{uid[:4]}_{num}",
            vals=f"'{uid}', '{num}'",
            num=num
        )


def generate_shell(count: int) -> Iterator[str]:
    """Generate safe shell commands."""
    for i in range(count):
        uid = f"{i:08x}"
        n = random.randint(1, 10000)
        
        fmt = random.randint(0, 19)
        if fmt == 0:
            yield f"ls -la {random.choice(PATHS)}/{uid}"
        elif fmt == 1:
            yield f"cat file_{uid}{random.choice(EXTENSIONS)}"
        elif fmt == 2:
            yield f"head -n {n} log_{uid}.txt"
        elif fmt == 3:
            yield f"tail -f /var/log/app_{uid}.log"
        elif fmt == 4:
            yield f"grep 'pattern_{uid}' data_{n}.txt"
        elif fmt == 5:
            yield f"find {random.choice(PATHS)} -name '*{uid}*'"
        elif fmt == 6:
            yield f"cp file_{uid}.txt backup_{uid}.txt"
        elif fmt == 7:
            yield f"mv old_{uid}.txt new_{uid}.txt"
        elif fmt == 8:
            yield f"mkdir -p {random.choice(PATHS)}/dir_{uid}"
        elif fmt == 9:
            yield f"chmod 644 file_{uid}.txt"
        elif fmt == 10:
            yield f"tar -czf archive_{uid}.tar.gz data_{uid}/"
        elif fmt == 11:
            yield f"curl -X GET https://api.example.com/v1/resource/{uid}"
        elif fmt == 12:
            yield f"wget https://cdn.example.com/files/{uid}.zip"
        elif fmt == 13:
            yield f"git clone https://github.com/user/repo-{uid}.git"
        elif fmt == 14:
            yield f"docker logs container_{uid}"
        elif fmt == 15:
            yield f"npm install package-{uid}"
        elif fmt == 16:
            yield f"pip install lib-{uid}=={random.randint(1,9)}.{random.randint(0,99)}.{random.randint(0,99)}"
        elif fmt == 17:
            yield f"python script_{uid}.py --arg={n}"
        elif fmt == 18:
            yield f"node app_{uid}.js --port={random.randint(3000,9000)}"
        else:
            yield f"echo 'task_{uid} completed with status {n}'"


def generate_api_calls(count: int) -> Iterator[str]:
    """Generate API call patterns."""
    for i in range(count):
        method = random.choice(METHODS)
        resource = random.choice(RESOURCES)
        rid = random.randint(1, 10000000)
        uid = f"{i:08x}"
        
        fmt = random.randint(0, 5)
        if fmt == 0:
            yield f'{{"method": "{method}", "url": "/api/v1/{resource}/{rid}", "request_id": "{uid}"}}'
        elif fmt == 1:
            yield f'{{"method": "{method}", "url": "/api/v{random.randint(1,3)}/{resource}?limit={random.randint(1,100)}&offset={rid}", "id": "{uid}"}}'
        elif fmt == 2:
            yield f'{{"method": "POST", "url": "/api/v1/{resource}", "body": {{"name": "item_{uid}", "value": {rid}}}}}'
        elif fmt == 3:
            yield f'{{"query": "{{ {resource}(id: {rid}) {{ id name createdAt }} }}", "operation": "{uid}"}}'
        elif fmt == 4:
            yield f'Authorization: Bearer token_{uid}_{rid}'
        else:
            yield f'X-Request-ID: {uid}-{rid}'


def generate_code_snippets(count: int) -> Iterator[str]:
    """Generate code snippets in various languages."""
    import uuid
    
    python_templates = [
        "def {func}_{uid}(x):\n    return x * {n}",
        "def {func}_{uid}({arg}):\n    return {arg} + {n}",
        "class {cls}_{uid}:\n    def __init__(self):\n        self.value = {n}",
        "class {cls}_{uid}:\n    def process(self, x):\n        return x * {n}",
        "import {module}\n{var}_{uid} = {module}.{func}()",
        "from {module} import {name}\nresult_{uid} = {name}({n})",
        "for i in range({n}):\n    print(f'item_{{i}}_{uid}')",
        "while {var}_{uid} < {n}:\n    {var}_{uid} += 1",
        "if {var}_{uid} > {n}:\n    result = {var}_{uid} * 2",
        "with open('{file}_{uid}.txt', 'r') as f:\n    data = f.read()",
        "try:\n    result_{uid} = func({n})\nexcept Exception as e:\n    print(e)",
        "[x * {n} for x in range({n2}) if x % 2 == 0]",
        "lambda x: x * {n} + {n2}",
        "data_{uid} = {{'key': {n}, 'value': '{val}'}}",
        "list_{uid} = [{n}, {n2}, {n3}]",
    ]
    
    js_templates = [
        "function {func}_{uid}(x) {{ return x * {n}; }}",
        "const {func}_{uid} = (x) => x * {n} + {n2};",
        "class {cls}_{uid} {{ constructor() {{ this.value = {n}; }} }}",
        "async function {func}_{uid}() {{ await fetch('/api/{n}'); }}",
        "const data_{uid} = await fetch('{url}/{n}');",
        "document.getElementById('{id}_{uid}').addEventListener('click', () => {{}});",
        "const arr_{uid} = [{n}, {n2}, {n3}];",
        "let obj_{uid} = {{ id: {n}, name: '{val}' }};",
        "export const {func}_{uid} = () => {n};",
        "import {{ {name} }} from '{module}';\nconst v_{uid} = {name}({n});",
    ]
    
    modules = ["os", "sys", "json", "re", "math", "datetime", "pathlib", "random", "collections", "itertools"]
    funcs = ["process", "handle", "compute", "validate", "transform", "parse", "load", "save", "get", "set"]
    classes = ["Handler", "Manager", "Service", "Controller", "Model", "View", "Factory", "Builder", "Adapter"]
    args = ["data", "value", "item", "obj", "config", "params", "options", "context"]
    
    for i in range(count):
        uid = f"{i:08x}"
        if random.random() > 0.5:
            t = random.choice(python_templates)
        else:
            t = random.choice(js_templates)
        
        yield t.format(
            func=random.choice(funcs),
            cls=random.choice(classes),
            module=random.choice(modules),
            name=random.choice(funcs),
            n=random.randint(1, 10000),
            n2=random.randint(1, 10000),
            n3=random.randint(1, 10000),
            var=random.choice(['x', 'y', 'value', 'count', 'idx']),
            arg=random.choice(args),
            file=f"data{random.randint(1,1000)}",
            url=f"https://api.example{random.randint(1,100)}.com",
            id=f"btn{random.randint(1,1000)}",
            val=''.join(random.choices(string.ascii_lowercase, k=8)),
            uid=uid
        )


def generate_logs(count: int) -> Iterator[str]:
    """Generate log entries in various formats."""
    messages = ["Request processed", "Connection established", "Cache hit",
                "User authenticated", "Task completed", "Data synced",
                "Session started", "File uploaded", "Query executed"]
    
    for _ in range(count):
        ts = datetime.now() - timedelta(seconds=random.randint(0, 86400))
        level = random.choice(LEVELS)
        service = random.choice(SERVICES)
        msg = random.choice(messages)
        ip = f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}"
        
        fmt = random.randint(0, 3)
        if fmt == 0:  # Apache-style
            yield f'{ip} - - [{ts.strftime("%d/%b/%Y:%H:%M:%S")}] "GET /api HTTP/1.1" 200 {random.randint(100,5000)}'
        elif fmt == 1:  # Syslog-style
            yield f'{ts.strftime("%b %d %H:%M:%S")} server {service}[{random.randint(1000,9999)}]: {msg}'
        elif fmt == 2:  # JSON
            yield f'{{"timestamp": "{ts.isoformat()}", "level": "{level}", "service": "{service}", "message": "{msg}"}}'
        else:  # Simple
            yield f'[{ts.strftime("%Y-%m-%d %H:%M:%S")}] [{level}] {service} - {msg}'


def generate_configs(count: int) -> Iterator[str]:
    """Generate configuration snippets."""
    for i in range(count):
        uid = f"{i:06x}"
        port = random.randint(3000, 65000)
        n = random.randint(1, 100)
        
        fmt = random.randint(0, 5)
        if fmt == 0:
            yield f"server:\n  host: localhost\n  port: {port}\n  id: {uid}"
        elif fmt == 1:
            yield f"database:\n  url: postgresql://localhost/db_{uid}\n  pool: {n}"
        elif fmt == 2:
            yield f'{{"name": "app-{uid}", "version": "{random.randint(1,9)}.{random.randint(0,99)}.{random.randint(0,99)}"}}'
        elif fmt == 3:
            yield f'{{"host": "localhost", "port": {port}, "id": "{uid}"}}'
        elif fmt == 4:
            yield f"DATABASE_URL=postgresql://localhost/db_{uid}"
        else:
            key = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
            yield f"API_KEY_{uid}={key}"


def generate_text(count: int) -> Iterator[str]:
    """Generate natural text content."""
    subjects = ["The system", "Our team", "The application", "This feature", "The update",
                "The server", "The database", "The API", "The service", "The module"]
    verbs = ["processes", "handles", "manages", "supports", "provides", "enables",
             "validates", "transforms", "optimizes", "monitors"]
    objects = ["user requests efficiently", "data in real-time", "multiple connections",
               "real-time updates", "secure transactions", "batch operations",
               "concurrent tasks", "async workflows", "streaming data", "cached results"]
    adjectives = ["quickly", "securely", "reliably", "efficiently", "seamlessly"]
    
    for i in range(count):
        uid = f"{i:06x}"
        parts = [
            f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)} {random.choice(adjectives)}.",
            f"Request {uid}: {random.choice(verbs)} completed in {random.randint(1,500)}ms.",
            f"Task {uid} {random.choice(verbs)} {random.randint(10,10000)} items.",
            f"User session {uid}: {random.choice(objects)}.",
            f"Process {uid} started at {random.randint(0,23):02d}:{random.randint(0,59):02d}.",
        ]
        yield random.choice(parts)
