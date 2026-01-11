"""Web scrapers for fresh real-world test data."""
import json
import random
import time
from datetime import datetime, timedelta
from typing import List, Iterator, Optional
from dataclasses import dataclass
import hashlib

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class ScrapedSample:
    text: str
    source: str
    timestamp: datetime
    category: str


class BaseScraper:
    """Base class for web scrapers."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session() if HAS_REQUESTS else None
        if self.session:
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (compatible; MLStressTest/1.0)'
            })
    
    def _get(self, url: str, params: dict = None) -> Optional[dict]:
        if not HAS_REQUESTS:
            return None
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Scraper error: {e}")
            return None
    
    def scrape(self, count: int = 100) -> List[ScrapedSample]:
        raise NotImplementedError


class WikipediaScraper(BaseScraper):
    """Scrape recent Wikipedia edits."""
    
    API_URL = "https://en.wikipedia.org/w/api.php"
    
    def scrape(self, count: int = 100) -> List[ScrapedSample]:
        samples = []
        params = {
            'action': 'query',
            'list': 'recentchanges',
            'rcprop': 'title|comment|timestamp',
            'rclimit': min(count, 500),
            'rctype': 'edit',
            'format': 'json',
        }
        
        data = self._get(self.API_URL, params)
        if not data:
            return samples
        
        for change in data.get('query', {}).get('recentchanges', []):
            title = change.get('title', '')
            comment = change.get('comment', '')
            
            if title:
                samples.append(ScrapedSample(
                    text=title,
                    source='wikipedia',
                    timestamp=datetime.utcnow(),
                    category='title'
                ))
            if comment:
                samples.append(ScrapedSample(
                    text=comment,
                    source='wikipedia',
                    timestamp=datetime.utcnow(),
                    category='comment'
                ))
        
        return samples[:count]


class RedditScraper(BaseScraper):
    """Scrape recent Reddit posts."""
    
    API_URL = "https://www.reddit.com/r/all/new.json"
    
    def scrape(self, count: int = 100) -> List[ScrapedSample]:
        samples = []
        params = {'limit': min(count, 100)}
        
        data = self._get(self.API_URL, params)
        if not data:
            return samples
        
        for post in data.get('data', {}).get('children', []):
            post_data = post.get('data', {})
            title = post_data.get('title', '')
            selftext = post_data.get('selftext', '')
            
            if title:
                samples.append(ScrapedSample(
                    text=title,
                    source='reddit',
                    timestamp=datetime.utcnow(),
                    category='title'
                ))
            if selftext and len(selftext) < 1000:
                samples.append(ScrapedSample(
                    text=selftext[:500],
                    source='reddit',
                    timestamp=datetime.utcnow(),
                    category='post'
                ))
        
        return samples[:count]


class HackerNewsScraper(BaseScraper):
    """Scrape Hacker News stories and comments."""
    
    API_URL = "https://hacker-news.firebaseio.com/v0"
    
    def scrape(self, count: int = 100) -> List[ScrapedSample]:
        samples = []
        
        # Get new story IDs
        data = self._get(f"{self.API_URL}/newstories.json")
        if not data:
            return samples
        
        story_ids = data[:min(count // 2, 50)]
        
        for story_id in story_ids:
            story = self._get(f"{self.API_URL}/item/{story_id}.json")
            if not story:
                continue
            
            title = story.get('title', '')
            if title:
                samples.append(ScrapedSample(
                    text=title,
                    source='hackernews',
                    timestamp=datetime.utcnow(),
                    category='title'
                ))
            
            # Get a few comments
            for kid_id in story.get('kids', [])[:3]:
                comment = self._get(f"{self.API_URL}/item/{kid_id}.json")
                if comment and comment.get('text'):
                    text = comment['text'][:500]
                    # Strip HTML tags simply
                    import re
                    text = re.sub(r'<[^>]+>', '', text)
                    samples.append(ScrapedSample(
                        text=text,
                        source='hackernews',
                        timestamp=datetime.utcnow(),
                        category='comment'
                    ))
            
            if len(samples) >= count:
                break
        
        return samples[:count]


class GitHubScraper(BaseScraper):
    """Scrape GitHub public events."""
    
    API_URL = "https://api.github.com/events"
    
    def scrape(self, count: int = 100) -> List[ScrapedSample]:
        samples = []
        
        data = self._get(self.API_URL)
        if not data:
            return samples
        
        for event in data:
            event_type = event.get('type', '')
            payload = event.get('payload', {})
            
            # Push events - commit messages
            if event_type == 'PushEvent':
                for commit in payload.get('commits', [])[:2]:
                    msg = commit.get('message', '')
                    if msg:
                        samples.append(ScrapedSample(
                            text=msg[:200],
                            source='github',
                            timestamp=datetime.utcnow(),
                            category='commit'
                        ))
            
            # Issue events
            elif event_type in ['IssuesEvent', 'IssueCommentEvent']:
                issue = payload.get('issue', {})
                title = issue.get('title', '')
                body = issue.get('body', '')
                
                if title:
                    samples.append(ScrapedSample(
                        text=title,
                        source='github',
                        timestamp=datetime.utcnow(),
                        category='issue_title'
                    ))
                if body and len(body) < 500:
                    samples.append(ScrapedSample(
                        text=body[:300],
                        source='github',
                        timestamp=datetime.utcnow(),
                        category='issue_body'
                    ))
            
            if len(samples) >= count:
                break
        
        return samples[:count]


class NewsScraper(BaseScraper):
    """Scrape news headlines from RSS feeds."""
    
    # Simple RSS parsing without external deps
    RSS_FEEDS = [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "http://rss.cnn.com/rss/edition.rss",
    ]
    
    def scrape(self, count: int = 100) -> List[ScrapedSample]:
        samples = []
        
        if not HAS_REQUESTS:
            return samples
        
        for feed_url in self.RSS_FEEDS:
            try:
                resp = self.session.get(feed_url, timeout=self.timeout)
                resp.raise_for_status()
                
                # Simple XML parsing for titles
                import re
                titles = re.findall(r'<title>(?:<!\[CDATA\[)?([^<\]]+)(?:\]\]>)?</title>', resp.text)
                
                for title in titles[1:]:  # Skip feed title
                    if title and len(title) > 10:
                        samples.append(ScrapedSample(
                            text=title.strip(),
                            source='news',
                            timestamp=datetime.utcnow(),
                            category='headline'
                        ))
                    
                    if len(samples) >= count:
                        break
            except Exception:
                continue
        
        return samples[:count]


class CVEScraper(BaseScraper):
    """Scrape recent CVE descriptions for malicious pattern testing."""
    
    API_URL = "https://cve.circl.lu/api/last"
    
    def scrape(self, count: int = 50) -> List[ScrapedSample]:
        samples = []
        
        data = self._get(self.API_URL)
        if not data:
            return samples
        
        for cve in data[:count]:
            summary = cve.get('summary', '')
            if summary:
                samples.append(ScrapedSample(
                    text=summary[:500],
                    source='cve',
                    timestamp=datetime.utcnow(),
                    category='vulnerability'
                ))
        
        return samples[:count]


class ScraperManager:
    """Manage all scrapers and combine results."""
    
    def __init__(self):
        self.scrapers = {
            'wikipedia': WikipediaScraper(),
            'reddit': RedditScraper(),
            'hackernews': HackerNewsScraper(),
            'github': GitHubScraper(),
            'news': NewsScraper(),
            'cve': CVEScraper(),
        }
    
    def scrape_all(self, count_per_source: int = 50) -> List[ScrapedSample]:
        """Scrape from all sources."""
        all_samples = []
        
        for name, scraper in self.scrapers.items():
            print(f"  Scraping {name}...")
            try:
                samples = scraper.scrape(count_per_source)
                all_samples.extend(samples)
                print(f"    Got {len(samples)} samples")
            except Exception as e:
                print(f"    Error: {e}")
            
            time.sleep(0.5)  # Rate limiting
        
        return all_samples
    
    def scrape_benign(self, count: int = 1000) -> List[str]:
        """Scrape benign samples only (no CVE)."""
        samples = []
        benign_scrapers = ['wikipedia', 'reddit', 'hackernews', 'github', 'news']
        
        per_source = count // len(benign_scrapers) + 1
        
        for name in benign_scrapers:
            scraper = self.scrapers[name]
            try:
                scraped = scraper.scrape(per_source)
                samples.extend([s.text for s in scraped])
            except Exception:
                continue
        
        random.shuffle(samples)
        return samples[:count]
    
    def scrape_for_stress_test(self, benign_count: int = 500, malicious_count: int = 100) -> dict:
        """Scrape samples for stress testing."""
        return {
            'benign': self.scrape_benign(benign_count),
            'malicious_context': [s.text for s in self.scrapers['cve'].scrape(malicious_count)],
        }


# Fallback generator when scraping fails
class FallbackGenerator:
    """Generate synthetic samples when scraping unavailable."""
    
    TEMPLATES = {
        'news': [
            "Breaking: {adj} {noun} reported in {place}",
            "{person} announces new {noun} initiative",
            "Study finds {adj} link between {noun} and {noun2}",
            "Markets {verb} amid {noun} concerns",
        ],
        'tech': [
            "New {lang} framework released for {purpose}",
            "How to implement {pattern} in {lang}",
            "Best practices for {noun} development",
            "{company} launches {adj} {product}",
        ],
        'social': [
            "Just finished {verb}ing my {noun}!",
            "Anyone else having issues with {noun}?",
            "TIL about {noun} and it's {adj}",
            "Check out this {adj} {noun} I found",
        ],
    }
    
    WORDS = {
        'adj': ['new', 'major', 'significant', 'unexpected', 'innovative', 'critical'],
        'noun': ['development', 'update', 'feature', 'system', 'project', 'release'],
        'noun2': ['performance', 'security', 'efficiency', 'growth', 'stability'],
        'place': ['New York', 'London', 'Tokyo', 'Berlin', 'Sydney', 'Toronto'],
        'person': ['CEO', 'Researcher', 'Developer', 'Analyst', 'Expert'],
        'verb': ['rise', 'fall', 'stabilize', 'surge', 'decline'],
        'lang': ['Python', 'JavaScript', 'Rust', 'Go', 'TypeScript'],
        'purpose': ['web development', 'data analysis', 'machine learning', 'automation'],
        'pattern': ['microservices', 'event sourcing', 'CQRS', 'clean architecture'],
        'company': ['Tech Corp', 'DataSoft', 'CloudNet', 'AI Systems'],
        'product': ['platform', 'service', 'tool', 'solution'],
    }
    
    def generate(self, count: int = 100) -> List[str]:
        samples = []
        categories = list(self.TEMPLATES.keys())
        
        for _ in range(count):
            category = random.choice(categories)
            template = random.choice(self.TEMPLATES[category])
            
            # Fill in template
            text = template
            for key, values in self.WORDS.items():
                while f'{{{key}}}' in text:
                    text = text.replace(f'{{{key}}}', random.choice(values), 1)
            
            samples.append(text)
        
        return samples


def get_fresh_samples(benign_count: int = 500, malicious_count: int = 100) -> dict:
    """Get fresh samples, with fallback if scraping fails."""
    manager = ScraperManager()
    
    try:
        result = manager.scrape_for_stress_test(benign_count, malicious_count)
        
        # If we didn't get enough, supplement with fallback
        if len(result['benign']) < benign_count // 2:
            fallback = FallbackGenerator()
            result['benign'].extend(fallback.generate(benign_count - len(result['benign'])))
        
        return result
    except Exception as e:
        print(f"Scraping failed, using fallback: {e}")
        fallback = FallbackGenerator()
        return {
            'benign': fallback.generate(benign_count),
            'malicious_context': [],
        }


if __name__ == "__main__":
    print("Testing scrapers...")
    
    manager = ScraperManager()
    samples = manager.scrape_all(count_per_source=10)
    
    print(f"\nTotal samples: {len(samples)}")
    for source in set(s.source for s in samples):
        count = len([s for s in samples if s.source == source])
        print(f"  {source}: {count}")
    
    print("\nSample texts:")
    for s in random.sample(samples, min(5, len(samples))):
        print(f"  [{s.source}] {s.text[:60]}...")
