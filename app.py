"""RELIC-IDS web app entry point.  Run: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="RELIC-IDS", page_icon=":material/shield:", layout="wide")

pages = [
    st.Page("pages/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("pages/detect.py", title="Detect", icon=":material/radar:"),
    st.Page("pages/investigate.py", title="Investigate alerts", icon=":material/manage_search:"),
    st.Page("pages/attack_lab.py", title="Attack lab", icon=":material/science:"),
    st.Page("pages/research.py", title="Research results", icon=":material/insights:"),
]
st.navigation(pages).run()
