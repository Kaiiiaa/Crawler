import streamlit as st
from modules import page_inspector, tree


st.set_page_config(page_title="AI draugelis seniems programuotojams", layout="wide")
st.sidebar.title("🧭 Nepasiklysk")

tool = st.sidebar.radio("Select Tool", [
    "Taxonomy builder",
    "Page Inspector",
])

if tool == "Taxonomy builder":
    tree.run()
elif tool == "Page Inspector":
    page_inspector.run()
