import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="Maps & Geospatial", page_icon="🗺️")
st.title("Maps and Geospatial Visualizations")
st.sidebar.header("Map Options")

# Generate sample geospatial data
@st.cache_data
def generate_geo_data():
    # Generate random points around the world
    np.random.seed(42)
    n_points = 1000
    
    # Bounds for the data (roughly world bounds)
    lat_bounds = (-60, 70)  # Avoiding extreme latitudes
    lon_bounds = (-180, 180)
    
    # Generate random points
    lats = np.random.uniform(lat_bounds[0], lat_bounds[1], n_points)
    lons = np.random.uniform(lon_bounds[0], lon_bounds[1], n_points)
    
    # Create categories and values
    categories = np.random.choice(['A', 'B', 'C', 'D'], n_points)
    values = np.random.normal(0, 1, n_points) * 10
    sizes = np.abs(values) + np.random.uniform(1, 5, n_points)
    
    # Create DataFrame
    df = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'category': categories,
        'value': values,
        'size': sizes
    })
    
    return df

geo_data = generate_geo_data()

# Map selection
map_type = st.sidebar.selectbox(
    "Select Map Type",
    ["Scatter Map", "Hexagon Map", "Heatmap", "3D Column Map", "Arc Map"]
)

map_style = st.sidebar.selectbox(
    "Select Map Style",
    ["Dark", "Light", "Satellite"]
)

style_dict = {
    "Dark": "mapbox://styles/mapbox/dark-v10",
    "Light": "mapbox://styles/mapbox/light-v10",
    "Satellite": "mapbox://styles/mapbox/satellite-v9"
}

st.subheader(f"{map_type} Example")
st.write("Using PyDeck for advanced geospatial visualizations")

# Initial view state
view_state = pdk.ViewState(
    latitude=0,
    longitude=0,
    zoom=1,
    pitch=0
)

if map_type == "Scatter Map":
    # Basic scatter map
    scatter_layer = pdk.Layer(
        'ScatterplotLayer',
        data=geo_data,
        get_position='[lon, lat]',
        get_color='[value > 0 ? 200 : 50, 100, value > 0 ? 50 : 200, 160]',
        get_radius='size * 10000',
        pickable=True,
        opacity=0.8,
    )
    
    deck = pdk.Deck(
        layers=[scatter_layer],
        initial_view_state=view_state,
        map_style=style_dict[map_style],
        tooltip={"text": "Value: {value}\nCategory: {category}"}
    )
    st.pydeck_chart(deck)

elif map_type == "Hexagon Map":
    hexagon_layer = pdk.Layer(
        'HexagonLayer',
        data=geo_data,
        get_position='[lon, lat]',
        radius=100000,
        elevation_scale=500,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,
        coverage=1,
        auto_highlight=True
    )
    
    deck = pdk.Deck(
        layers=[hexagon_layer],
        initial_view_state=view_state,
        map_style=style_dict[map_style],
    )
    st.pydeck_chart(deck)

elif map_type == "Heatmap":
    heatmap_layer = pdk.Layer(
        'HeatmapLayer',
        data=geo_data,
        get_position='[lon, lat]',
        opacity=0.9,
        get_weight='size',
        aggregation='"MEAN"',
        threshold=0.05
    )
    
    deck = pdk.Deck(
        layers=[heatmap_layer],
        initial_view_state=view_state,
        map_style=style_dict[map_style],
    )
    st.pydeck_chart(deck)

elif map_type == "3D Column Map":
    column_layer = pdk.Layer(
        'ColumnLayer',
        data=geo_data,
        get_position='[lon, lat]',
        get_elevation='size * 1000',
        elevation_scale=100,
        radius=50000,
        get_fill_color='[255 * (value > 0 ? 1 : 0), 100, 255 * (value < 0 ? 1 : 0), 140]',
        pickable=True,
        auto_highlight=True,
        extruded=True,
    )
    
    # Use a different view state with pitch for 3D visualization
    view_state_3d = pdk.ViewState(
        latitude=0,
        longitude=0,
        zoom=1,
        pitch=45
    )
    
    deck = pdk.Deck(
        layers=[column_layer],
        initial_view_state=view_state_3d,
        map_style=style_dict[map_style],
        tooltip={"text": "Value: {value}\nSize: {size}"}
    )
    st.pydeck_chart(deck)

elif map_type == "Arc Map":
    # For the Arc Map, generate some origin-destination pairs
    np.random.seed(42)
    n_arcs = 100
    
    # Generate random points for origins
    origins = geo_data.sample(n_arcs)[['lon', 'lat', 'value']]
    origins = origins.rename(columns={'lon': 'lon_origin', 'lat': 'lat_origin'})
    
    # Generate random points for destinations
    destinations = geo_data.sample(n_arcs)[['lon', 'lat']]
    destinations = destinations.rename(columns={'lon': 'lon_dest', 'lat': 'lat_dest'})
    
    # Combine to create arc data
    arc_data = pd.concat([origins.reset_index(drop=True), 
                         destinations.reset_index(drop=True)], axis=1)
    arc_data['value'] = np.abs(arc_data['value'])
    
    arc_layer = pdk.Layer(
        'ArcLayer',
        data=arc_data,
        get_source_position='[lon_origin, lat_origin]',
        get_target_position='[lon_dest, lat_dest]',
        get_width='value * 2',
        get_source_color=[255, 0, 0, 200],
        get_target_color=[0, 0, 255, 200],
        pickable=True,
    )
    
    deck = pdk.Deck(
        layers=[arc_layer],
        initial_view_state=view_state,
        map_style=style_dict[map_style],
    )
    st.pydeck_chart(deck)

# Show filtered data
st.subheader("Data Explorer")

# Category filter
categories = sorted(geo_data['category'].unique())
selected_categories = st.multiselect("Filter by Category", categories, default=categories[:2])

if selected_categories:
    filtered_data = geo_data[geo_data['category'].isin(selected_categories)]
    
    # Show map of filtered data
    view_state_filtered = pdk.ViewState(
        latitude=filtered_data['lat'].mean(),
        longitude=filtered_data['lon'].mean(),
        zoom=1
    )
    
    filtered_layer = pdk.Layer(
        'ScatterplotLayer',
        data=filtered_data,
        get_position='[lon, lat]',
        get_color='[200, 30, 100, 160]',
        get_radius='size * 5000',
        pickable=True
    )
    
    filtered_deck = pdk.Deck(
        layers=[filtered_layer],
        initial_view_state=view_state_filtered,
        map_style=style_dict[map_style],
        tooltip={"text": "Value: {value}\nCategory: {category}"}
    )
    
    st.pydeck_chart(filtered_deck)
    
    # Show data table
    with st.expander("Show Data Table"):
        st.dataframe(filtered_data)
else:
    st.warning("Please select at least one category.")
