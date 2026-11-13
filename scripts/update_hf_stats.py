#!/usr/bin/env python3
"""Render zhangdw's lifetime HF downloads using public Hub data only.

downloadsAllTime is cumulative; downloads is a rolling 30-day count and must
never be used here. Both models and datasets contribute to the single total.
Run from any directory: python3 scripts/update_hf_stats.py
"""

import json
import re
import sys
import time
from html import escape
from pathlib import Path
from string import Template
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree


ACCOUNT = "zhangdw"
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("huggingface-card.svg")
OUTPUT_DIR = ROOT / "profile-stats"
PALETTES = {
    "huggingface-panel.svg": {
        "background": "#ffffff",
        "border": "#d1d9e0",
        "heading": "#24292f",
        "muted": "#59636e",
    },
    "huggingface-panel-dark.svg": {
        "background": "#0d1117",
        "border": "#3d444d",
        "heading": "#e6edf3",
        "muted": "#9198a1",
    },
}


def fetch_page(url):
    """Retry transient Hub failures, but never replace unavailable counts by 0."""
    for attempt in range(3):
        request = Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "zhangdw156-github-profile/1.0",
        })
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response), response.headers.get("Link", "")
        except HTTPError as error:
            if error.code != 429 and not 500 <= error.code < 600:
                raise
            if attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def total_downloads():
    total = 0
    for kind in ("models", "datasets"):
        query = urlencode({
            "author": ACCOUNT,
            "limit": 1000,
            "expand[]": "downloadsAllTime",
        })
        url = f"https://huggingface.co/api/{kind}?{query}"
        pages_seen = set()
        repos_seen = set()
        while url:
            parsed = urlsplit(url)
            if (parsed.scheme, parsed.netloc, parsed.path) != (
                "https", "huggingface.co", f"/api/{kind}"
            ):
                raise ValueError("Unexpected Hub pagination URL")
            if url in pages_seen:
                raise ValueError("Repeated Hub pagination URL")
            pages_seen.add(url)
            repos, links = fetch_page(url)
            if not isinstance(repos, list):
                raise ValueError(f"Invalid {kind} response from Hugging Face")
            for repo in repos:
                repo_id = repo.get("id", "") if isinstance(repo, dict) else ""
                if not isinstance(repo_id, str) or not repo_id.startswith(f"{ACCOUNT}/"):
                    raise ValueError(f"Unexpected repository in {kind} response")
                count = repo.get("downloadsAllTime")
                if type(count) is not int or count < 0:
                    raise ValueError(f"Missing or invalid lifetime downloads for {repo_id}")
                if repo_id in repos_seen:
                    raise ValueError(f"Repeated repository in {kind} response: {repo_id}")
                repos_seen.add(repo_id)
                total += count
            next_page = re.search(r'<([^>]+)>;\s*rel="next"', links)
            url = next_page.group(1) if next_page else None
    return total


def update_cards(destination=OUTPUT_DIR):
    # Fetch and validate everything before touching either previously good card.
    total = total_downloads()
    template = Template(TEMPLATE.read_text(encoding="utf-8"))
    cards = {}
    for filename, palette in PALETTES.items():
        svg = template.substitute(
            **palette, account=escape(ACCOUNT), downloads=f"{total:,}"
        )
        ElementTree.fromstring(svg)
        cards[filename] = svg
    destination.mkdir(parents=True, exist_ok=True)
    for filename, svg in cards.items():
        (destination / filename).write_text(svg, encoding="utf-8")
    print(f"HF lifetime downloads: {total:,}")
    return total


if __name__ == "__main__":
    try:
        update_cards()
    except (OSError, ValueError, ElementTree.ParseError) as error:
        print(f"HF stats update failed: {error}", file=sys.stderr)
        sys.exit(1)
