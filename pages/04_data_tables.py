import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Data Tables", page_icon="📊")
st.title("Advanced Data Tables")
st.sidebar.header("Table Options")

# Generate sample data
@st.cache_data
def generate_table_data():
    np.random.seed(42)
    
    # Create dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    
    # Create product data
    products = ["Widget A", "Widget B", "Widget C", "Gadget X", "Gadget Y"]
    categories = ["Electronics", "Home Goods", "Office Supplies"]
    regions = ["North", "South", "East", "West"]
    
    # Create sample data
    data = {
        "date": np.random.choice(dates, 500),
        "product": np.random.choice(products, 500),
        "category": np.random.choice(categories, 500),
        "region": np.random.choice(regions, 500),
        "sales": np.random.randint(10, 1000, 500),
        "quantity": np.random.randint(1, 50, 500),
        "customer_rating": np.random.uniform(1, 5, 500).round(1),
        "in_stock": np.random.choice([True, False], 500, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    df["revenue"] = df["sales"] * df["quantity"]
    df["profit_margin"] = np.random.uniform(0.1, 0.4, 500).round(2)
    df["profit"] = (df["revenue"] * df["profit_margin"]).round(2)
    df["date"] = pd.to_datetime(df["date"])
    
    return df

df = generate_table_data()

# Table display options
table_type = st.sidebar.selectbox(
    "Select Table Type",
    ["Basic Table", "Styled Table", "Interactive Table", "Aggregated Table"]
)

st.subheader(f"{table_type} Example")

if table_type == "Basic Table":
    st.write("Basic Table using st.dataframe")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.write("Table with column configuration")
    st.dataframe(
        df.head(10),
        column_config={
            "product": "Product Name",
            "sales": st.column_config.NumberColumn("Sales ($)", format="$%d"),
            "customer_rating": st.column_config.NumberColumn("Rating", format="%.1f ⭐"),
            "in_stock": st.column_config.CheckboxColumn("Available"),
            "profit_margin": st.column_config.ProgressColumn("Margin", format="%.0f%%", min_value=0, max_value=0.4)
        },
        use_container_width=True
    )

elif table_type == "Styled Table":
    st.write("Styled Table with highlighting")
    
    # Get a subset of data to display
    display_df = df.head(15).copy()
    
    # Create a styled version of the dataframe
    def highlight_high_profit(val):
        """Highlight high profit values in green"""
        if isinstance(val, (int, float)):
            if val > df["profit"].quantile(0.75):
                return 'background-color: rgba(0, 255, 0, 0.2)'
        return ''
    
    def highlight_low_stock(row):
        """Highlight low stock rows in red"""
        if not row["in_stock"]:
            return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
        return [''] * len(row)
    
    # Apply styles
    styled_df = display_df.style.applymap(highlight_high_profit, subset=["profit"])\
                               .apply(highlight_low_stock, axis=1)
    
    st.dataframe(styled_df, use_container_width=True)
    
    st.write("Bar Chart Table")
    # Create a bar chart in the table for profit margin
    bar_df = df.head(10).copy()
    st.dataframe(
        bar_df,
        column_config={
            "profit_margin": st.column_config.ProgressColumn(
                "Profit Margin",
                help="The profit margin percentage",
                format="%.1f%%",
                min_value=0,
                max_value=0.4,
            ),
            "customer_rating": st.column_config.BarChartColumn(
                "Rating Distribution",
                help="Distribution of customer ratings",
                y_min=1,
                y_max=5,
            ),
        },
        hide_index=True,
        use_container_width=True
    )

elif table_type == "Interactive Table":
    st.write("Interactive Table with Filtering and Sorting")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_categories = st.multiselect(
            "Filter by Category",
            df["category"].unique(),
            default=df["category"].unique()
        )
    with col2:
        selected_regions = st.multiselect(
            "Filter by Region",
            df["region"].unique(),
            default=df["region"].unique()
        )
    with col3:
        min_sales, max_sales = st.slider(
            "Sales Range",
            int(df["sales"].min()),
            int(df["sales"].max()),
            (int(df["sales"].min()), int(df["sales"].max()))
        )
    
    # Apply filters
    filtered_df = df[
        (df["category"].isin(selected_categories)) &
        (df["region"].isin(selected_regions)) &
        (df["sales"] >= min_sales) &
        (df["sales"] <= max_sales)
    ]
    
    # Show the filtered data
    st.write(f"Showing {len(filtered_df)} of {len(df)} records")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Add download button
    st.download_button(
        "Download Filtered Data as CSV",
        filtered_df.to_csv(index=False).encode('utf-8'),
        "filtered_data.csv",
        "text/csv",
        key='download-csv'
    )

elif table_type == "Aggregated Table":
    st.write("Aggregated Table with Summary Statistics")
    
    # Select dimensions for aggregation
    agg_dimension = st.selectbox(
        "Group By",
        ["category", "region", "product", "date"]
    )
    
    if agg_dimension == "date":
        # For date, we'll group by month
        df["month"] = df["date"].dt.strftime('%Y-%m')
        agg_df = df.groupby("month").agg({
            "sales": "sum",
            "quantity": "sum",
            "revenue": "sum",
            "profit": "sum",
            "customer_rating": "mean"
        }).reset_index()
        agg_df["customer_rating"] = agg_df["customer_rating"].round(2)
    else:
        agg_df = df.groupby(agg_dimension).agg({
            "sales": "sum",
            "quantity": "sum",
            "revenue": "sum",
            "profit": "sum",
            "customer_rating": "mean"
        }).reset_index()
        agg_df["customer_rating"] = agg_df["customer_rating"].round(2)
    
    # Sort by revenue by default
    agg_df = agg_df.sort_values("revenue", ascending=False)
    
    # Display the aggregated data
    st.dataframe(
        agg_df,
        column_config={
            "sales": st.column_config.NumberColumn("Total Sales", format="$%d"),
            "revenue": st.column_config.NumberColumn("Total Revenue", format="$%d"),
            "profit": st.column_config.NumberColumn("Total Profit", format="$%.2f"),
            "customer_rating": st.column_config.NumberColumn("Avg Rating", format="%.2f ⭐")
        },
        use_container_width=True
    )
    
    # Add summary metrics
    st.subheader("Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"${df['revenue'].sum():,.0f}")
    with col2:
        st.metric("Total Profit", f"${df['profit'].sum():,.2f}")
    with col3:
        st.metric("Average Rating", f"{df['customer_rating'].mean():.2f} ⭐")
    with col4:
        st.metric("In Stock %", f"{(df['in_stock'].sum() / len(df) * 100):.1f}%")

# Show raw data option
with st.expander("Show Raw Data"):
    st.dataframe(df, use_container_width=True)
