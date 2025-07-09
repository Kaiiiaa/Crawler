import streamlit as st
import pandas as pd
import json
from datetime import datetime
from rag_graph import create_graph
from modules.memory import (
    lookup_summary,
    store_summary,
    list_cached_urls,
    get_all_documents,
)
import tiktoken
import os

def count_tokens(text: str, model: str = "gpt-4") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4") -> float:
    return (prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.06)

def run():
    st.title("🧠 Agentic Page Inspector (LangGraph RAG + Memory + Export)")

    url = st.text_input("Enter URL to analyze:", "")

    if st.button("🧹 Clear Memory Store"):
        import shutil
        shutil.rmtree("memory_store", ignore_errors=True)
        st.warning("Memory cleared!")

    # 👁 View existing memory entries
    st.subheader("📚 View Cached Summaries")
    docs = get_all_documents()
    if docs["metadatas"]:
        options = [d["url"] for d in docs["metadatas"]]
        selected_url = st.selectbox("Choose a cached URL:", options)

        if selected_url:
            cached = lookup_summary(selected_url)
            if cached:
                st.markdown(f"**Summary for** `{selected_url}`")
                st.text_area("🧠 Summary", cached["summary"], height=250)
                st.markdown("### 🧪 Inspection Notes")
                for note in cached.get("inspection_notes", []):
                    st.markdown(f"- {note}")

                # Export JSON
                st.download_button(
                    "⬇️ Export as JSON",
                    data=json.dumps(cached, indent=2),
                    file_name="summary.json",
                    mime="application/json"
                )

                # Export Markdown
                md = f"""# Summary for {selected_url}

**Status:** {cached.get("status")}  
**Redirects:** {cached.get("redirects")}  
**Timestamp:** {cached.get("timestamp")}

## Summary
{cached['summary']}

## Inspection Notes
""" + "\n".join(f"- {n}" for n in cached.get("inspection_notes", []))

                st.download_button(
                    "⬇️ Export as Markdown",
                    data=md,
                    file_name="summary.md",
                    mime="text/markdown"
                )

    st.divider()

    # Main run button
    if st.button("🚀 Run LangGraph Agent") and url:

        with st.spinner("🔎 Checking memory..."):
            cached = lookup_summary(url)

        if cached:
            st.success("✅ Loaded from memory")
            st.text_area("🧠 Summary (cached)", cached.get("summary", ""), height=300)
            return

        # Run the actual LangGraph
        with st.spinner("⚙️ Running agentic inspection..."):
            graph = create_graph()
            result = graph.invoke({"url": url})
            if "scrapability_score" in result:
                score = result["scrapability_score"]
                st.subheader("📊 Scrapability Score")
                if score >= 80:
                    st.success(f"✅ Score: {score} — Good for scraping!")
                elif score >= 50:
                    st.warning(f"⚠️ Score: {score} — Partial access likely.")
                else:
                    st.error(f"🚫 Score: {score} — Site may block or obfuscate scraping.")

        # Manual token estimation
        html = result.get("html", "")
        context = result.get("context", "")
        notes = "\n".join(result.get("inspection_notes", []))
        summary_text = result.get("summary", "")

        prompt = f"""Inspection Findings:\n{notes}\n\nContext:\n{context}\n\nHTML:\n{html}"""
        prompt_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(summary_text)
        total_tokens = prompt_tokens + completion_tokens
        cost = estimate_cost(prompt_tokens, completion_tokens)

        structured = {
            "url": url,
            "summary": summary_text,
            "inspection_notes": result.get("inspection_notes", []),
            "scrapability_score": result.get("scrapability_score"),
            "issue_details": result.get("issue_details"),
            "status": result.get("status"),
            "redirects": result.get("redirects"),
            "timestamp": datetime.now().isoformat(),
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens,
                "cost": cost,
            }
        }

        store_summary(url, structured)

        st.subheader("🔍 Final Summary")
        st.text_area("🧠 AI Insight", summary_text, height=300)

        st.markdown("### 📊 Token Usage & Cost")
        st.markdown(f"- Prompt: {prompt_tokens}")
        st.markdown(f"- Completion: {completion_tokens}")
        st.markdown(f"- Total: {total_tokens}")
        st.markdown(f"- Estimated Cost: ${cost:.4f}")
        st.success("📦 Result stored in memory")
