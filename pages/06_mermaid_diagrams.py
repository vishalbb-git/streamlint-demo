import streamlit as st
import tempfile
import subprocess
import os
import base64
from streamlit_ace import st_ace

st.set_page_config(layout="wide")

st.title("🔀 Mermaid Diagram Generator")

st.markdown("""
Generate various types of diagrams using Mermaid syntax. You can either type directly or load a .mmd file.
""")

st.subheader("More about Mermaid")
st.markdown("[Visit Mermaid Documentation](https://mermaid.js.org/intro/)")

# Add file uploader
uploaded_file = st.file_uploader("Upload a Mermaid file (.mmd)", type=['mmd'])


# Initialize or update mermaid_code based on file upload
initial_value = ""
output_filename = "diagram"  # default filename

if uploaded_file:
    initial_value = uploaded_file.getvalue().decode()
    # Get original filename without extension
    output_filename = os.path.splitext(uploaded_file.name)[0]

st.info("💡 You can customzie the loaded Mermaid input file further below")
mermaid_code = st_ace(
    value=initial_value,
    height=400,
    theme="github",
    show_gutter=True,
)

if st.button("Generate SVG"):
    with tempfile.NamedTemporaryFile(suffix='.mmd', mode='w', delete=False) as f:
        f.write(mermaid_code)
        temp_mmd = f.name
        
    temp_svg = temp_mmd.replace('.mmd', '.svg')
    
    st.warning("Note: To generate SVG files, you need to have mermaid-cli installed. Run: `npm install -g @mermaid-js/mermaid-cli`")       

    try:
        # Assuming mmdc (mermaid-cli) is installed
        subprocess.run(['mmdc', '-i', temp_mmd, '-o', temp_svg], check=True)
            
        with open(temp_svg, 'r') as f:
            svg_content = f.read()
            
            # Create download link for SVG
            b64 = base64.b64encode(svg_content.encode()).decode()
            href = f'<div style="text-align: right;"><a download="{output_filename}.svg" href="data:image/svg+xml;base64,{b64}">Download SVG</a></div>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Display SVG directly without zoom controls
            st.markdown(svg_content, unsafe_allow_html=True)
            
        # Cleanup temp files
        os.unlink(temp_mmd)
        os.unlink(temp_svg)
    except subprocess.CalledProcessError:
        st.error("Error generating SVG. Please make sure mermaid-cli is installed.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")



