import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

st.set_page_config(page_title="Basic Charts", page_icon="📊")
st.title("Basic Charts and Plots")
st.sidebar.header("Basic Charts Options")

# Generate sample data
@st.cache_data
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
    data = pd.DataFrame({
        "date": dates,
        "value": np.random.randn(30).cumsum(),
        "category": np.random.choice(["A", "B", "C"], 30),
        "size": np.random.randint(10, 100, 30)
    })
    return data

data = generate_data()

# Chart selection
chart_type = st.sidebar.selectbox(
    "Select Chart Type",
    ["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot", "Histogram", "Altair"]
)

st.subheader(f"{chart_type} Example")
st.write("Using sample data with 30 data points")

# Display different chart types
if chart_type == "Line Chart":
    st.line_chart(data.set_index("date")["value"])
    st.code("""st.line_chart(data.set_index("date")["value"])""")
    
elif chart_type == "Bar Chart":
    st.bar_chart(data.set_index("date")["value"])
    st.code("""st.bar_chart(data.set_index("date")["value"])""")
    
elif chart_type == "Area Chart":
    st.area_chart(data.set_index("date")["value"])
    st.code("""st.area_chart(data.set_index("date")["value"])""")
    
elif chart_type == "Scatter Plot":
    fig, ax = plt.subplots()
    ax.scatter(data["value"], data["size"], c=pd.factorize(data["category"])[0], alpha=0.7)
    ax.set_xlabel("Value")
    ax.set_ylabel("Size")
    ax.set_title("Scatter Plot by Category")
    st.pyplot(fig)
    st.code("""
    fig, ax = plt.subplots()
    ax.scatter(data["value"], data["size"], c=pd.factorize(data["category"])[0], alpha=0.7)
    st.pyplot(fig)
    """)
    
elif chart_type == "Histogram":
    fig, ax = plt.subplots()
    ax.hist(data["value"], bins=10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Values")
    st.pyplot(fig)
    st.code("""
    fig, ax = plt.subplots()
    ax.hist(data["value"], bins=10)
    st.pyplot(fig)
    """)
    
elif chart_type == "Altair":
    chart = alt.Chart(data).mark_circle().encode(
        x='date',
        y='value',
        color='category',
        size='size'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.code("""
    chart = alt.Chart(data).mark_circle().encode(
        x='date',
        y='value',
        color='category',
        size='size'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
    """)

# Show data
with st.expander("Show Sample Data"):
    st.dataframe(data)
