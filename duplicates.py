import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import io
import os

# Set page configuration with dark theme
st.set_page_config(layout="wide", page_title="Duplicate Item Explorer", page_icon="👕",
                   initial_sidebar_state="collapsed")

# Apply styling with focused white background for item cards
st.markdown("""
<style>
    /* App background */
    .stApp {
        background-color: #0e1117;
    }

    /* Create white background containers for items */
    .item-container {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
        width: 100%;
        display: block;
    }

    /* Force images to expand to fill container width */
    .item-container img {
        width: 100%;
        height: auto;
        object-fit: contain;
    }

    /* Text color for light backgrounds */
    .item-container a, .item-container p {
        color: #333 !important;
    }

    /* Text color for dark backgrounds */
    body, .stText, h1, h2, h3 {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Performance optimization settings
st.cache_data.clear()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None

if 'item_images' not in st.session_state:
    st.session_state.item_images = {}

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None


# Function to create a placeholder image when image can't be loaded
def create_placeholder_html(item_id):
    """Create HTML for a placeholder with white background"""
    color_r = (hash(str(item_id)) * 50) % 256
    color_g = (hash(str(item_id)) * 30) % 256
    color_b = (hash(str(item_id)) * 70) % 256

    color_hex = f"#{color_r:02x}{color_g:02x}{color_b:02x}"

    html = f"""
    <div style="width: 300px; height: 400px; background-color: white; display: flex; align-items: center; justify-content: center; margin: 0 auto;">
        <div style="width: 100%; height: 100%; background-color: white; display: flex; align-items: center; justify-content: center; color: white; font-size: 24px;">
            {item_id}
        </div>
    </div>
    """

    return html


# Function to create a placeholder image
def create_placeholder_image(item_id):
    """Create a placeholder image with the item ID displayed on it"""
    # Using smaller dimensions to minimize white space
    width, height = 200, 280  # Keeping approx 5:7 ratio but smaller
    background_color = (255, 255, 255)  # White background
    text_color = (50, 50, 50)  # Dark gray text

    # Create a new image with white background
    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)

    # Try to use a font, or fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    # Draw the item ID in the center
    text = str(item_id)

    # Handle different PIL versions for text size calculation
    if hasattr(draw, 'textsize'):
        textwidth, textheight = draw.textsize(text, font=font)
    elif hasattr(draw, 'textlength'):
        # For newer PIL versions
        textwidth = draw.textlength(text, font=font)
        textheight = font.getsize(text)[1] if hasattr(font, 'getsize') else 24
    else:
        # Fallback estimations
        textwidth, textheight = 100, 24

    x = (width - textwidth) // 2
    y = (height - textheight) // 2

    # Draw the text
    draw.text((x, y), text, font=font, fill=text_color)

    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return img_byte_arr.getvalue()


# Function to fetch images but NOT process them - just return the raw bytes
@st.cache_data(ttl=3600)
def fetch_images_batch(image_urls_dict):
    """Fetch multiple images in parallel and return as a dictionary of raw image bytes"""
    import concurrent.futures
    import threading

    # Create a thread-safe dictionary to store results
    results = {}
    results_lock = threading.Lock()

    def fetch_single_image(key, url):
        """Fetch a single image without processing and add to results"""
        try:
            if not url or not isinstance(url, str) or url == 'nan' or url == '':
                return

            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                # Just store the raw bytes - no processing
                with results_lock:
                    results[key] = response.content
        except Exception:
            # Silently fail
            pass

    # Process images in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for key, url in image_urls_dict.items():
            futures.append(executor.submit(fetch_single_image, key, url))

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    return results


# Attempt to load data from upload or local file
if st.session_state.df is None:
    st.header("Upload Dataset")

    # File uploader
    uploaded_file = st.file_uploader("Upload duplicates CSV file", type=['csv'],
                                     help="Upload a CSV file with duplicate item clusters")

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        try:
            with st.spinner("Loading uploaded data..."):
                # Use cache_data to speed up loading
                @st.cache_data(ttl=3600)
                def load_uploaded_csv(file):
                    return pd.read_csv(file)


                df = load_uploaded_csv(uploaded_file)

                # Validate the required columns
                required_columns = ["id", "Outfit Item ID", "Outfit Item Cover Link", "duplicate_clusters"]
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    st.error(f"Missing required columns in uploaded file: {', '.join(missing_columns)}")
                    st.info("Required columns: id, Outfit Item ID, Outfit Item Cover Link, duplicate_clusters")
                    st.stop()

                # Store in session state
                st.session_state.df = df

                # Success message
                st.success(f"Dataset loaded successfully with {len(df)} items!")
                st.rerun()
        except Exception as e:
            st.error(f"Error loading uploaded data: {e}")
            st.exception(e)
            st.stop()

    # Alternative: Try to load from local file if no upload
    st.markdown("---")
    st.markdown("### Or use local file")

    if st.button("Load from local file (duplicates.csv)"):
        try:
            file_path = "duplicates.csv"
            with st.spinner(f"Loading data from {file_path}..."):
                @st.cache_data(ttl=3600)
                def load_csv(path):
                    return pd.read_csv(path)


                df = load_csv(file_path)

                # Validate the required columns
                required_columns = ["id", "Outfit Item ID", "Outfit Item Cover Link", "duplicate_clusters"]
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    st.error(f"Missing required columns in {file_path}: {', '.join(missing_columns)}")
                    st.stop()

                # Store in session state
                st.session_state.df = df

                # Success message
                st.success(f"Dataset loaded successfully with {len(df)} items!")
                st.rerun()
        except FileNotFoundError:
            st.error(f"File 'duplicates.csv' not found in the current directory.")
            st.info("Please make sure the file exists in the same directory as this application.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.exception(e)
            st.stop()

    # Display sample CSV format and instructions
    with st.expander("CSV Format Requirements", expanded=False):
        st.markdown("""
        ### Required CSV Format

        Your CSV file must contain the following columns:

        - `id`: Unique identifier for each item
        - `Outfit Item ID`: ID used for admin panel links
        - `Outfit Item Cover Link`: URL to the item image
        - `duplicate_clusters`: Cluster ID grouping similar items

        ### Sample CSV Format:
        ```
        id,Outfit Item ID,Outfit Item Cover Link,duplicate_clusters
        1001,a1b2c3d4,https://example.com/image1.jpg,101
        1002,e5f6g7h8,https://example.com/image2.jpg,101
        1003,i9j0k1l2,https://example.com/image3.jpg,102
        ```
        """)

    st.stop()  # Stop execution until file is loaded

# Main Interface with cluster selection and item display
if st.session_state.df is not None:
    # Create a two-column layout
    left_col, right_col = st.columns([1, 3])

    # Left column with cluster selection
    with left_col:
        st.header("Duplicate Clusters")


        # Get unique cluster IDs and their sizes - use caching for better performance
        @st.cache_data(ttl=600)
        def get_cluster_statistics(df):
            cluster_sizes = df.groupby('duplicate_clusters').size()
            return cluster_sizes.sort_values(ascending=False)


        cluster_sizes = get_cluster_statistics(st.session_state.df)

        # Display statistics
        st.subheader("Statistics")
        st.info(f"Total potential duplicates: {len(st.session_state.df)}")
        st.info(f"Total duplicate groups: {len(cluster_sizes)}")

        # Add a two-sided slider to filter clusters by size
        max_cluster_size = cluster_sizes.max()
        min_cluster_size = 2
        max_selectable_size = max(max_cluster_size, min_cluster_size)

        size_filter_range = st.slider(
            "Filter groups by number of duplicates:",
            min_value=min_cluster_size,
            max_value=max_selectable_size,
            value=(min_cluster_size, max_selectable_size),
            step=1
        )

        min_size_filter, max_size_filter = size_filter_range

        # Filter clusters by the selected size range (keeping the sort by size)
        filtered_clusters_with_sizes = cluster_sizes[(cluster_sizes >= min_size_filter) &
                                                     (cluster_sizes <= max_size_filter)]

        st.info(
            f"Showing {len(filtered_clusters_with_sizes)} groups with {min_size_filter}-{max_size_filter} duplicates")

        # Search functionality for clusters
        search_term = st.text_input("🔍 Search by duplicate group ID", "")

        # Fix for search functionality to avoid errors
        if search_term:
            try:
                search_value = int(search_term)
                # Check if this cluster exists in our dataset
                if search_value in filtered_clusters_with_sizes.index:
                    selected_cluster = search_value
                else:
                    st.warning(f"Group {search_value} not found or doesn't meet size criteria")
                    selected_cluster = None
            except ValueError:
                st.warning("Please enter a valid duplicate group ID number")
                selected_cluster = None
        else:
            # No search term - use selectbox for filtered clusters
            if not filtered_clusters_with_sizes.empty:
                # Create cluster options
                options_data = [(int(idx), int(size)) for idx, size in filtered_clusters_with_sizes.items()]

                # Create display options
                display_options = [f"Cluster {cluster} ({size} items)" for cluster, size in options_data]

                # Preselect the first cluster by default
                default_index = 0 if display_options else None

                # Display the dropdown
                selected_option = st.selectbox(
                    "Select a duplicate group:",
                    options=display_options,
                    index=default_index,
                    key="cluster_selector"
                )

                if selected_option:
                    # Extract cluster ID - simplified parsing for better performance
                    try:
                        # Parse the cluster ID from the option string (format: "Cluster {cluster} ({size} items)")
                        cluster_str = selected_option.split(' ')[1]  # Get cluster number
                        selected_cluster = int(cluster_str)
                    except:
                        # Fallback to index-based lookup
                        idx = display_options.index(selected_option)
                        if idx >= 0 and idx < len(options_data):
                            selected_cluster = options_data[idx][0]
                        else:
                            selected_cluster = None
                else:
                    selected_cluster = None
            else:
                st.warning("No clusters meet the size criteria")
                selected_cluster = None

        # Update the session state
        st.session_state.selected_cluster = selected_cluster

    # Right column with the item display
    with right_col:
        # If selected_cluster is None but we have filtered clusters, preselect the first one
        if st.session_state.selected_cluster is None and not filtered_clusters_with_sizes.empty:
            first_cluster = filtered_clusters_with_sizes.index[0]
            st.session_state.selected_cluster = int(first_cluster)
            st.rerun()

        if st.session_state.selected_cluster is not None:
            cluster_id = st.session_state.selected_cluster

            # Use a container for all content to prevent jittering
            main_container = st.container()

            with main_container:
                # Simple header - no background
                st.header(f"Items in Cluster #{cluster_id}")


                # Pre-process the data for this cluster - use caching for better performance
                @st.cache_data(ttl=300)
                def get_cluster_items(df, cluster_id):
                    # Use native pandas filtering for speed
                    filtered_df = df[df['duplicate_clusters'] == cluster_id]
                    # Convert only necessary columns to optimize memory
                    items = filtered_df[['id', 'Outfit Item ID', 'Outfit Item Cover Link']].to_dict('records')
                    return items, len(items)


                # Get the items and count
                cluster_items, item_count = get_cluster_items(st.session_state.df, cluster_id)

                # Display number of items in this cluster
                st.text(f"Found {item_count} items in this cluster")

                # Display all items on a single page since there are at most 50 items per group
                page_items = cluster_items

                # Show the total number of items (without "Showing all X items" text)
                # st.text(f"Showing all {item_count} items in this cluster")

                # Fetch all images for the current page at once (parallel processing)
                with st.spinner("Loading images..."):
                    # Prepare a dictionary with item_id -> image_url mapping
                    image_urls_dict = {item['id']: item['Outfit Item Cover Link'] for item in page_items}

                    # Fetch all images in parallel
                    fetched_images = fetch_images_batch(image_urls_dict)

                # Add global CSS for consistent grid layout
                st.markdown("""
                <style>
                .item-grid {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 16px;
                    margin-bottom: 20px;
                }
                .item-card {
                    background: white;
                    border: 1px solid #333;
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                }
                .image-container {
                    width: 200px;
                    margin: 0 auto;
                    text-align: center;
                    padding: 0;
                }
                .image-container img {
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                }
                .item-info {
                    padding: 5px;
                    display: flex;
                    flex-direction: column;
                    background-color: white;
                }
                </style>
                """, unsafe_allow_html=True)

                # Calculate grid layout - using 3 columns per row
                cols_per_row = 3

                # Create rows manually to avoid index issues
                for i in range(0, len(page_items), cols_per_row):
                    # Create row with columns
                    cols = st.columns(cols_per_row)

                    # Process items in this row
                    for j in range(cols_per_row):
                        idx = i + j

                        # Skip if we're out of items
                        if idx >= len(page_items):
                            continue

                        # Get item data
                        item = page_items[idx]

                        # Display in the appropriate column - simplified for performance
                        with cols[j]:
                            try:
                                # Extract item information
                                item_id = item['id']
                                outfit_id = item['Outfit Item ID']

                                # Use the pre-fetched image or a placeholder
                                if item_id in fetched_images:
                                    img = fetched_images[item_id]
                                else:
                                    # Use placeholder with consistent dimensions
                                    img = create_placeholder_image(item_id)

                                # Create a container with fixed dimensions for consistent layout
                                with st.container():
                                    # Prepare base64 encoded image for HTML display
                                    import base64

                                    image_base64 = base64.b64encode(img).decode() if isinstance(img, bytes) else ""

                                    # Add CSS to enforce consistent image container height
                                    st.markdown(f"""
                                    <style>
                                    .fixed-height-container {{
                                        height: 300px;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        overflow: hidden;
                                        background-color: white;
                                    }}
                                    .fixed-height-container img {{
                                        max-height: 300px;
                                        width: auto !important;
                                        max-width: 100%;
                                        object-fit: contain;
                                    }}
                                    </style>

                                    <div class="fixed-height-container">
                                        <img src="data:image/jpeg;base64,{image_base64}" alt="Item #{item_id}" />
                                    </div>
                                    """, unsafe_allow_html=True)

                                    # Format the admin link correctly with query parameter
                                    admin_url = f"https://chic-control.com/outfit-item/?id={outfit_id}"

                                    # Display the link
                                    st.markdown(f"[View in Admin Panel]({admin_url})")

                                    # Add copyable text for convenience
                                    st.code(outfit_id, language="text")
                            except Exception as e:
                                # Minimal error handling for speed
                                st.error(f"Error: {str(e)}")
        else:
            # No cluster selected - show overview
            st.header("Select a Cluster")
            st.info("Please select a duplicate cluster from the left panel to view its items.")

            # Show general statistics
            st.subheader("Dataset Overview")

            # Count clusters by size
            cluster_size_counts = cluster_sizes.value_counts().sort_index()

            # Create a summary dataframe
            summary_data = {
                "Cluster Size": cluster_size_counts.index.tolist(),
                "Count": cluster_size_counts.values.tolist()
            }
            summary_df = pd.DataFrame(summary_data)

            # Display the summary
            st.write("Number of clusters by size:")
            st.dataframe(summary_df)