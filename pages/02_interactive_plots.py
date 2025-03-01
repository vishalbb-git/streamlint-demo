import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Interactive Plots", page_icon="📈")
st.title("Interactive Data Visualizations")
st.sidebar.header("Interactive Plot Options")

# Generate sample data
@st.cache_data
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    categories = ["Technology", "Healthcare", "Finance", "Retail", "Energy"]
    regions = ["North", "South", "East", "West", "Central"]
    
    df = pd.DataFrame({
        "date": np.repeat(dates, 5),
        "value": np.random.normal(0, 1, 500).cumsum(),
        "category": np.tile(np.repeat(categories, 1), 100),
        "region": np.random.choice(regions, 500),
        "volume": np.random.randint(100, 1000, 500)
    })
    return df

data = generate_data()

# Plot selection
plot_type = st.sidebar.selectbox(
    "Select Plot Type",
    ["Line Plot", "Bar Plot", "Scatter Plot", "3D Scatter", "Heatmap", "Choropleth"]
)

st.subheader(f"{plot_type} Example")
st.write("Interactive plots created with Plotly")

if plot_type == "Line Plot":
    aggregated = data.groupby(['date', 'category'])['value'].mean().reset_index()
    fig = px.line(aggregated, x="date", y="value", color="category", 
                  title="Trends by Category", line_shape="spline")
    st.plotly_chart(fig, use_container_width=True)
    
elif plot_type == "Bar Plot":
    aggregated = data.groupby('category')['value'].sum().reset_index()
    fig = px.bar(aggregated, x="category", y="value",
                color="category", title="Total Value by Category")
    st.plotly_chart(fig, use_container_width=True)
    
elif plot_type == "Scatter Plot":
    fig = px.scatter(data, x="value", y="volume", color="category", 
                   size="volume", hover_data=["region"],
                   title="Scatter Plot with Size and Color")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "3D Scatter":
    # Create some 3D data
    np.random.seed(42)
    z_data = np.random.randn(100)
    fig = px.scatter_3d(data.iloc[:100], x="value", y="volume", z=z_data,
                      color="category", size="volume", 
                      title="3D Scatter Visualization")
    st.plotly_chart(fig, use_container_width=True)
    
elif plot_type == "Heatmap":
    pivot = pd.pivot_table(data, values="value", index="date", columns="category", aggfunc="mean")
    recent_pivot = pivot.tail(20)  # Last 20 days
    fig = px.imshow(recent_pivot, title="Value Heatmap by Category (Last 20 Days)",
                   labels=dict(x="Category", y="Date", color="Value"),
                   color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)
    
elif plot_type == "Choropleth":
    st.write("This would typically use geographical data. Using a simple example with US states:")
    
    # Simple mockup of US state data
    us_states = {
        'state': ['California', 'Texas', 'Florida', 'New York', 'Pennsylvania',
                 'Illinois', 'Ohio', 'Georgia', 'Michigan', 'North Carolina'],
        'code': ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'MI', 'NC'],
        'value': np.random.randint(20, 100, 10)
    }
    state_df = pd.DataFrame(us_states)
    
    fig = px.choropleth(state_df, 
                       locations='code', 
                       locationmode='USA-states', 
                       scope='usa',
                       color='value',
                       color_continuous_scale='Viridis',
                       title='US States Value Distribution')
    st.plotly_chart(fig, use_container_width=True)

# Interactive controls
st.subheader("Interactive Controls")

# Filter by category
selected_categories = st.multiselect(
    "Filter by Category", 
    options=data["category"].unique(),
    default=data["category"].unique()[:2]
)

if selected_categories:
    filtered_data = data[data["category"].isin(selected_categories)]
    
    # Create a custom interactive plot based on selection
    custom_fig = px.scatter(filtered_data, x="date", y="value", 
                          color="category", size="volume", 
                          title=f"Custom Interactive Plot for {', '.join(selected_categories)}")
    st.plotly_chart(custom_fig, use_container_width=True)
else:
    st.warning("Please select at least one category.")

# Show data
with st.expander("Show Sample Data"):
    st.dataframe(data.head(20))
