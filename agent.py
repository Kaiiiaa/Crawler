# agent.py

import os
import re
from urllib.parse import urljoin, urlparse

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from bs4 import BeautifulSoup
import requests
import gzip
import io
from lxml import etree
from playwright.sync_api import sync_playwright

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@tool
def crawl_tool(url: str) -> str:
    """Extracts links from the given webpage using DOM scraping"""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000, wait_until='domcontentloaded')
            content = page.content()
            browser.close()

        soup = BeautifulSoup(content, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            text = a.get_text(strip=True)
            if href and text and len(text) > 2:
                links.append(f"{text}: {href}")
        return "\n".join(links[:50]) or "No links found."
    except Exception as e:
        return f"DOM crawl failed: {str(e)}"


@tool
def embedded_json_tool(url: str) -> str:
    """Fetch embedded JSON (like navigation/category data) from a webpage"""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000, wait_until='domcontentloaded')
            content = page.content()
            browser.close()

        soup = BeautifulSoup(content, "html.parser")
        for script in soup.find_all("script"):
            if script.string and "{" in script.string:
                json_start = script.string.find("{")
                json_text = script.string[json_start:]
                if len(json_text) > 200:
                    return json_text[:1500]  
        return "No JSON found."
    except Exception as e:
        return f"Failed to extract JSON: {str(e)}"


@tool
def sitemap_tool(domain: str) -> str:
    """Parses sitemap.xml or sitemap_index.xml from a domain and returns category-like links"""
    try:
        robots_url = urljoin(domain, "/robots.txt")
        r = requests.get(robots_url, timeout=10)
        sitemap_urls = re.findall(r"Sitemap:\s*(\S+)", r.text)

        results = []
        for sitemap_url in sitemap_urls:
            urls = parse_sitemap(sitemap_url)
            for link in urls:
                if re.search(r"/(category|cp|departments|browse)/", link):
                    results.append(link)
            if results:
                break
        return "\n".join(results[:50]) or "No category links found."
    except Exception as e:
        return f"Sitemap extraction failed: {str(e)}"


def parse_sitemap(url, visited=None):
    if visited is None:
        visited = set()
    if url in visited:
        return []
    visited.add(url)

    urls = []
    try:
        if url.endswith(".gz"):
            r = requests.get(url, timeout=10)
            with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as f:
                content = f.read()
        else:
            r = requests.get(url, timeout=10)
            content = r.content

        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(content, parser=parser)
        ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        sitemap_tags = root.xpath("//ns:sitemap/ns:loc", namespaces=ns)
        if sitemap_tags:
            for sub in sitemap_tags:
                urls.extend(parse_sitemap(sub.text.strip(), visited))
        else:
            for loc in root.xpath("//ns:url/ns:loc", namespaces=ns):
                urls.append(loc.text.strip())
    except:
        pass
    return urls


def run_agent_task(start_url: str, callbacks=None) -> str:
    print("🧪 Inside run_agent_task. Callbacks:", callbacks)

    llm = ChatOpenAI(model="gpt-4", temperature=0, callbacks=callbacks or [])
    print("🧪 LLM initialized")

    tools = [crawl_tool, embedded_json_tool, sitemap_tool]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        callbacks=callbacks or [],
        verbose=True,
    )

    parsed = urlparse(start_url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    objective = f"""You are a helpful AI assistant. 
    Your goal is to find valid product or category navigation links from the website: {start_url}.
    Use the tools provided to collect real category links like Electronics, Home Decor, Health, etc.
    Ignore links like Login, Blog, Cart, Terms, etc."""

    result = agent_executor.run(objective)

    print("🧪 Agent finished. Returning result.")
    return result
