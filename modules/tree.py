import streamlit as st
from urllib.parse import urlparse, urljoin
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from langchain.callbacks import OpenAICallbackHandler
from contextlib import contextmanager
from agent import run_agent_task

MAX_LINKS_PER_PAGE = 150
MAX_DEPTH = 3

def run():
    st.title("🧠 AI-Powered Homepage Category Extractor")

    start_url = st.text_input("Enter homepage URL:", "")
    use_agent = st.checkbox("🤖 Use LangChain AI Agent", value=True)
    show_tree = st.checkbox("🌲 Show raw link hierarchy tree", value=True)

    if not start_url:
        return

    parsed = urlparse(start_url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    visited = set()
    raw_links = {}
    category_tree = {}
    hierarchy_levels = {}
    link_sources = {}

    with st.spinner("🔍 Crawling raw links..."):
        category_tree = crawl(start_url, domain, visited, raw_links, link_sources, hierarchy_levels, parent="Root", depth=0)

    if raw_links:
        st.subheader("🔗 Raw Links Found")
        st.json(raw_links)

        # Download Raw Links
        raw_df = pd.DataFrame([
            {"Label": label, "URL": url, "Source": link_sources.get(label, ""), "Hierarchy Level": hierarchy_levels.get(label, "")}
            for label, url in raw_links.items()
        ])
        st.download_button("⬇️ Download Raw Links", raw_df.to_csv(index=False), "raw_links.csv", "text/csv")

    final_links = {}

    if use_agent:
        st.subheader("🤖 Running LangChain Agent...")
        with st.spinner("Agent working..."):
            handler = OpenAICallbackHandler()
            result = run_agent_task(start_url, callbacks=[handler])

            st.text_area("🧠 Agent Output", result, height=300)

            # 🧠 Capture result into final_links
            try:
                final_links = eval(result) if result.strip().startswith("{") else {}
            except Exception as e:
                st.warning(f"⚠️ Could not parse agent output: {e}")
                final_links = {}

            # 📊 Token usage
            if handler.total_tokens > 0:
                st.markdown("### 📊 Token Usage & Cost")
                st.markdown(f"- **Prompt Tokens:** {handler.prompt_tokens}")
                st.markdown(f"- **Completion Tokens:** {handler.completion_tokens}")
                st.markdown(f"- **Total Tokens:** {handler.total_tokens}")
                st.markdown(f"- **Estimated Cost (USD):** ${handler.total_cost:.4f}")
            else:
                st.warning("⚠️ No token usage detected. Was the agent run correctly?")


    if final_links:
        st.success("✅ Filtered Category Links")
        st.json(final_links)

        filtered_df = pd.DataFrame([
            {"Label": label, "URL": url}
            for label, url in final_links.items()
        ])
        st.download_button("⬇️ Download Filtered Links", filtered_df.to_csv(index=False), "filtered_links.csv", "text/csv")

    if show_tree and category_tree:
        st.subheader("🌳 Category Tree View")
        display_tree(category_tree)

def crawl(url, domain, visited, raw_links, link_sources, hierarchy_levels, parent, depth):
    if url in visited or len(visited) > MAX_LINKS_PER_PAGE or depth > MAX_DEPTH:
        return {}

    visited.add(url)
    tree = {}

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        content = response.content

        soup = BeautifulSoup(content, "html.parser")
        count = 0
        for a in soup.find_all("a", href=True):
            if count > MAX_LINKS_PER_PAGE:
                break

            href = urljoin(url, a["href"])
            label = a.get_text(strip=True)

            if not is_internal_link(href, domain) or not label:
                continue

            raw_links[label] = href
            link_sources[label] = "dom"
            hierarchy_levels[label] = f"Level {depth}"
            tree.setdefault(parent, {})[label] = href

            if re.search(r"/(category|departments|browse|cp|c)/", href):
                subtree = crawl(href, domain, visited, raw_links, link_sources, hierarchy_levels, parent=label, depth=depth+1)
                if subtree:
                    tree.update(subtree)

            count += 1

    except Exception as e:
        st.warning(f"Failed to crawl {url}: {e}")

    return tree


def is_internal_link(href, domain):
    return domain in href and not href.startswith("#") and not href.startswith("javascript")

def display_tree(tree, level=0):
    indent = "    " * level
    for parent, children in tree.items():
        st.markdown(f"{indent}**{parent}**")
        for label, url in children.items():
            st.markdown(f"{indent}- [{label}]({url})")

print("🧪 After run_agent_task:")
