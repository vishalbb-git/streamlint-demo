import streamlit as st

st.set_page_config(
    page_title="Streamlit Visualization Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎨 Streamlit Visualization Gallery")
st.markdown("""
This demo application showcases various visualization capabilities using Streamlit.
Navigate through the pages in the sidebar to explore different visualization types.
""")

st.header("Welcome to the Demo")

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What you'll find")
        st.write("""
        - Basic charts and plots
        - Interactive data visualizations
        - Maps and geospatial data
        - Advanced data tables
        - Custom components and layouts
        """)
    
    with col2:
        st.subheader("Libraries Featured")
        st.write("""
        - Plotly
        - Matplotlib
        - Altair
        - PyDeck
        - Pandas
        - Streamlit Native Components
        """)

st.info("👈 Select a demo from the sidebar to get started!")
