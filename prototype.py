import streamlit as st
import pandas as pd
import numpy as np
import random
import json
from PIL import Image, ImageDraw
import io
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import requests
import time

# Set page configuration
st.set_page_config(layout="wide", page_title="Fashion Recommender System")

# Add CSS to force the columns to be sticky
st.markdown("""
<style>
    /* Target the specific column elements with more precise selectors */
    div[data-testid="stHorizontalBlock"] > div:first-child {
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 3rem;
        align-self: flex-start;
        height: calc(100vh - 5rem);
        overflow-y: auto;
    }

    div[data-testid="stHorizontalBlock"] > div:last-child {
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 3rem;
        align-self: flex-start;
        height: calc(100vh - 5rem);
        overflow-y: auto;
    }

    /* Ensure the middle column can scroll independently */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
        overflow-y: auto;
        max-height: none;
    }

    /* Ensure headers stay at the top of their column */
    .sticky-header {
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 0;
        background-color: white;
        z-index: 999;
        padding: 1rem 0;
    }

    /* Add some extra spacing for the elements */
    .element-container {
        margin-bottom: 1rem;
    }

    /* Make the main container take full height */
    .main .block-container {
        max-width: 100%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 0;
    }

    /* Button styling */
    div.row-widget.stButton > button {
        margin-right: 5px;
        display: inline-block;
    }

    /* Make buttons display side by side */
    .stButton {
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

if 'outfit_images' not in st.session_state:
    st.session_state.outfit_images = {}


# Define all functions first, before they're used

# Function to convert user preferences to a downloadable JSON file
def get_downloadable_preferences_json():
    """
    Converts the current user preferences to a JSON string for download
    """
    # Create a simple dictionary with user preferences and timestamp
    preferences_data = {
        "user_preferences": st.session_state.user_preferences,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "liked_outfits": [outfit_id for outfit_id, interaction in st.session_state.outfit_interactions.items()
                          if interaction == 'like'],
        "disliked_outfits": [outfit_id for outfit_id, interaction in st.session_state.outfit_interactions.items()
                             if interaction == 'dislike']
    }

    # Convert to JSON string
    json_str = json.dumps(preferences_data, indent=4)

    return json_str


# Function to create a download button for user preferences
def create_download_button_for_preferences():
    """
    Creates a download button for the user preferences JSON
    """
    # Only create the button if preferences exist
    if hasattr(st.session_state, 'user_preferences') and st.session_state.user_preferences:
        # Get the preferences as JSON
        json_str = get_downloadable_preferences_json()

        # Create a download button
        st.download_button(
            label="Download Preferences as JSON",
            data=json_str,
            file_name=f"fashion_preferences_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_preferences_button",
            help="Download your current preferences and liked/disliked items as a JSON file"
        )


# Function to create a horizontal bar chart for preferences
@st.cache_data
def create_preference_chart(preferences, category, is_user_preferences=True):
    """Create a preference chart - cacheable function"""
    items = list(preferences.keys())
    values = list(preferences.values())

    fig, ax = plt.subplots(figsize=(8, max(3, len(items) * 0.4)))

    # Create horizontal bar chart
    bars = ax.barh(items, values, height=0.5)

    # Color bars based on value
    for i, bar in enumerate(bars):
        if is_user_preferences:
            # For user preferences (-1 to 1): Red for negative, blue for positive
            if values[i] < 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        else:
            # For outfit features (0 to 1): Gradient of blue intensity
            intensity = 0.3 + 0.7 * values[i]  # Scale to avoid too light colors
            bar.set_color((0, 0, intensity))

    # Add values to the end of each bar
    for i, v in enumerate(values):
        offset = 0.05 if v >= 0 else -0.05
        ax.text(v + offset, i, f"{v:.1f}",
                va='center', fontsize=10,
                color='black')

    # Set title and labels
    ax.set_title(f"{category}")

    if is_user_preferences:
        # For user preferences: -1 to 1 range
        ax.set_xlim(-1.1, 1.1)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)  # Add vertical line at x=0
    else:
        # For outfit features: 0 to 1 range
        ax.set_xlim(0, 1.1)

    plt.tight_layout()

    # Convert plot to image
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return buf


# Function to load and parse outfit data from Parquet
@st.cache_data
def load_outfit_data(file):
    # Load the parquet file
    df = pd.read_parquet(file)

    outfits = []

    # Pre-process the dataframe to ensure dictionaries are properly formatted
    for col in ['color_scores', 'textile_scores', 'pattern_scores']:  # Fixed column name
        if col in df.columns:
            # Convert dictionary strings to actual dictionaries if needed
            if df[col].dtype == 'object' and isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(lambda x: json.loads(x.replace("'", "\"")) if isinstance(x, str) else x)

    # Parse each row into our outfit structure
    for i, row in df.iterrows():
        try:
            # Create outfit metadata structure - using clean data from preprocessing
            outfit_metadata = {
                'Color': row['color_scores'],
                'Textile': row['textile_scores'],
                'Visual patterns': row['pattern_scores']  # Fixed column name but kept category name
            }

            # Create outfit object
            outfit = {
                'id': i,
                'name': f"Outfit {i + 1}",
                'metadata': outfit_metadata,
                'image_base64': row.get('image', None),
                'image_url': row.get('image_link', None)
            }

            outfits.append(outfit)

        except Exception as e:
            st.warning(f"Error parsing row {i}: {e}")
            continue

    # Extract all unique features across outfits to initialize user preferences
    features = {
        'Color': set(),
        'Textile': set(),
        'Visual patterns': set()
    }

    for outfit in outfits:
        for category, values in outfit['metadata'].items():
            features[category].update(values.keys())

    # Convert sets to lists
    for category in features:
        features[category] = list(features[category])

    return outfits, features


# Optimized image processing functions
@st.cache_data
def process_image_from_base64(base64_str, outfit_id):
    """Process base64 string to image - cacheable function"""
    try:
        # Clean up the base64 string if needed
        if isinstance(base64_str, str) and ',' in base64_str:
            base64_str = base64_str.split(',')[1]

        # Decode base64 to image
        img_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_data))
        return img
    except:
        return None


@st.cache_data
def process_image_from_url(image_url, outfit_id):
    """Process image URL - cacheable function"""
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None


# Function to generate a placeholder image
@st.cache_data
def create_placeholder_image(outfit_id):
    """Create a placeholder image - cacheable function"""
    # Create a colored background based on outfit_id
    color_r = (outfit_id * 50) % 256
    color_g = (outfit_id * 30) % 256
    color_b = (outfit_id * 70) % 256

    # Create a PIL Image with the colored background
    img = Image.new('RGB', (300, 400), (color_r, color_g, color_b))

    # Draw outfit number
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)

    # Try to use a system font or fallback to default
    try:
        font = ImageFont.truetype("Arial", 40)
    except IOError:
        font = ImageFont.load_default()

    draw.text((100, 180), f"Outfit {outfit_id + 1}", fill=(255, 255, 255), font=font)
    return img


def get_outfit_image(outfit, liked=None):
    """Optimized function to get outfit image with like/dislike indicator"""
    outfit_id = outfit['id']

    # Check if we already have a processed image for this outfit (without like/dislike indicator)
    if outfit_id not in st.session_state.outfit_images:
        # Try base64 first
        img = None
        if outfit.get('image_base64') and outfit['image_base64'] != 'nan' and outfit['image_base64'] is not None:
            img = process_image_from_base64(outfit['image_base64'], outfit_id)

        # Try URL next if base64 failed
        if img is None and outfit.get('image_url') and outfit['image_url'] != 'nan' and outfit['image_url'] is not None:
            img = process_image_from_url(outfit['image_url'], outfit_id)

        # Fallback to placeholder if both failed
        if img is None:
            img = create_placeholder_image(outfit_id)

        # Store the base image without like/dislike indicator
        st.session_state.outfit_images[outfit_id] = img
    else:
        # Use the cached image
        img = st.session_state.outfit_images[outfit_id]

    # For performance, if no like/dislike indicator needed, return the image as is
    if liked is None:
        # Convert to bytes for display
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr

    # Make a copy of the image to avoid modifying the cached one
    img_copy = img.copy()

    # Add like/dislike indicator if needed
    draw = ImageDraw.Draw(img_copy)
    indicator_color = (0, 255, 0) if liked else (255, 0, 0)
    draw.ellipse((img_copy.width - 30, 10, img_copy.width - 10, 30), fill=indicator_color)

    # Convert to bytes for display
    img_byte_arr = BytesIO()
    img_copy.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr


# Function to update user preferences based on outfit interaction
def update_preferences(outfit_metadata, interaction_weight):
    learning_rate = st.session_state.learning_rate  # η (eta) from the formula

    for category, features in outfit_metadata.items():
        for feature, feature_value in features.items():  # v_f (feature value in outfit)
            if category in st.session_state.user_preferences and feature in st.session_state.user_preferences[category]:
                # Get current user preference (u_f^old)
                current_preference = st.session_state.user_preferences[category][feature]

                # Apply the update formula: u_f^new = u_f^old + η * W * v_f
                new_preference = current_preference + learning_rate * interaction_weight * feature_value

                # Clamp the preference between -1 and 1
                st.session_state.user_preferences[category][feature] = max(min(new_preference, 1.0), -1.0)


# Function to handle outfit interaction (like/dislike) - no longer displays metadata
def handle_interaction(outfit_id, interaction):
    previous = st.session_state.outfit_interactions.get(outfit_id, None)

    # Define interaction weights
    like_weight = 1  # W = 1 for like
    dislike_weight = -1  # W = -1 for dislike

    # If the same button is clicked again, toggle the interaction off
    if previous == interaction:
        del st.session_state.outfit_interactions[outfit_id]
        # Reverse the previous preference update with opposite weight
        weight_to_apply = -like_weight if previous == 'like' else -dislike_weight
        update_preferences(st.session_state.outfits[outfit_id]['metadata'], weight_to_apply)
    else:
        # If changing from like to dislike or vice versa, first reverse the previous update
        if previous is not None:
            weight_to_reverse = -like_weight if previous == 'like' else -dislike_weight
            update_preferences(st.session_state.outfits[outfit_id]['metadata'], weight_to_reverse)

        # Set the new interaction
        st.session_state.outfit_interactions[outfit_id] = interaction

        # Update preferences based on the new interaction
        weight_to_apply = like_weight if interaction == 'like' else dislike_weight
        update_preferences(st.session_state.outfits[outfit_id]['metadata'], weight_to_apply)


# Function to select an outfit to view its details
def select_outfit(outfit_id):
    st.session_state.selected_outfit = outfit_id


# Lazy loading optimization for preference visualization
@st.cache_data(ttl=1)  # Cache for just 1 second to avoid recomputing on every interaction
def get_user_preference_charts(user_preferences):
    """Pre-render all user preference charts at once"""
    charts = {}
    for category, preferences in user_preferences.items():
        if preferences:
            charts[category] = create_preference_chart(preferences, category, is_user_preferences=True)
    return charts


# Format outfit metadata as text
def format_metadata_as_text(metadata):
    """Format outfit metadata as text with feature values"""
    formatted_text = ""
    for category, features in metadata.items():
        formatted_text += f"### {category}\n"
        # Sort features by value (descending)
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        for feature, value in sorted_features:
            if value > 0:  # Only show features with non-zero values
                formatted_text += f"- **{feature}**: {value:.1f}\n"
        formatted_text += "\n"
    return formatted_text


# Load data if not already loaded
if not st.session_state.file_uploaded:
    # File uploader
    uploaded_file = st.file_uploader("Upload outfits file", type=['parquet'])

    if uploaded_file is not None:
        try:
            # Load outfits and extract features
            outfits, features = load_outfit_data(uploaded_file)

            if not outfits:
                st.error("No valid outfits found in the file.")
                st.stop()

            # Initialize user preferences to zero for all features
            user_preferences = {}
            for category, feature_list in features.items():
                user_preferences[category] = {feature: 0.0 for feature in feature_list}

            # Save to session state
            st.session_state.outfits = outfits
            st.session_state.features = features
            st.session_state.user_preferences = user_preferences
            st.session_state.outfit_interactions = {}
            st.session_state.selected_outfit = None
            st.session_state.learning_rate = 0.2
            st.session_state.file_uploaded = True

            st.success(f"Outfit data loaded successfully! Found {len(outfits)} outfits.")
            st.rerun()  # Rerun to show the main interface
        except Exception as e:
            st.error(f"Error loading the parquet file: {e}")
            st.exception(e)  # This will show the full traceback

    # Add option to use default data if no file is uploaded
    if st.button("Use Sample Data"):
        try:
            # Load sample features
            features = {
                'Color': ['red', 'blue', 'white', 'pink', 'black', 'green', 'yellow'],
                'Textile': ['denim', 'silk', 'cotton', 'polyester', 'wool', 'leather'],
                'Visual patterns': ['floral', 'text', 'imagery', 'stripes', 'plaid', 'solid']
            }

            # Generate sample outfits
            outfits = []
            for i in range(20):
                outfit_metadata = {}

                for category, feature_list in features.items():
                    outfit_metadata[category] = {}
                    active_features = random.sample(feature_list, random.randint(1, 3))

                    for feature in feature_list:
                        if feature in active_features:
                            outfit_metadata[category][feature] = round(random.uniform(0.3, 1.0), 1)
                        else:
                            outfit_metadata[category][feature] = 0.0

                outfit = {
                    'id': i,
                    'name': f"Outfit {i + 1}",
                    'metadata': outfit_metadata,
                    'image_base64': None,
                    'image_url': None
                }
                outfits.append(outfit)

            # Initialize user preferences to zero
            user_preferences = {}
            for category, feature_list in features.items():
                user_preferences[category] = {feature: 0.0 for feature in feature_list}

            # Save to session state
            st.session_state.outfits = outfits
            st.session_state.features = features
            st.session_state.user_preferences = user_preferences
            st.session_state.outfit_interactions = {}
            st.session_state.selected_outfit = None
            st.session_state.learning_rate = 0.2
            st.session_state.file_uploaded = True

            st.success("Sample data loaded successfully!")
            st.rerun()  # Rerun to show the main interface
        except Exception as e:
            st.error(f"Error generating sample data: {e}")
            st.exception(e)  # This will show the full traceback

    # Stop execution here if data is not loaded
    st.stop()

# Main application interface
st.title("Fashion Recommender System Prototype")

# Create a three-column layout
left_col, middle_col, right_col = st.columns([1, 2, 1])

with left_col:
    # Sticky header
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    st.header("User Preferences")

    # Add a slider to adjust learning rate
    st.session_state.learning_rate = st.slider(
        "Learning Rate (η)",
        min_value=0.1,
        max_value=0.5,
        value=st.session_state.learning_rate,
        step=0.05,
        help="Controls how quickly preferences update (higher values mean faster changes)"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Optimization: Batch render all preference charts at once
    preference_charts = get_user_preference_charts(st.session_state.user_preferences)
    for category, chart_img in preference_charts.items():
        st.image(chart_img, use_container_width=True)

    # Add a separator before the download button
    st.markdown("---")
    st.subheader("Export Preferences")

    # Add the download button
    create_download_button_for_preferences()

with middle_col:
    # Sticky header for middle column
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    st.header("Outfit Collection")
    st.markdown('</div>', unsafe_allow_html=True)

    # Display outfits in a grid (2 per row)
    rows = [st.container() for _ in range((len(st.session_state.outfits) + 1) // 2)]

    for i, outfit in enumerate(st.session_state.outfits):
        row_idx = i // 2
        col_idx = i % 2

        # Create a horizontal layout for this row if it's the first item in the row
        if col_idx == 0:
            with rows[row_idx]:
                outfit_cols = st.columns(2)

        with outfit_cols[col_idx]:
            outfit_id = outfit['id']

            # Get the interaction status for this outfit
            interaction = st.session_state.outfit_interactions.get(outfit_id, None)

            # Display the outfit image
            img = get_outfit_image(outfit, liked=True if interaction == 'like' else (
                False if interaction == 'dislike' else None))
            st.image(img, caption=outfit['name'], use_container_width=True)

            # Create buttons WITHOUT columns, just placed side by side using CSS styling
            like_btn = "👍 Liked" if interaction == 'like' else "👍"
            if st.button(like_btn, key=f"like_{outfit_id}"):
                handle_interaction(outfit_id, 'like')
                st.rerun()

            dislike_btn = "👎 Disliked" if interaction == 'dislike' else "👎"
            if st.button(dislike_btn, key=f"dislike_{outfit_id}"):
                handle_interaction(outfit_id, 'dislike')
                st.rerun()

            if st.button("Details", key=f"details_{outfit_id}"):
                select_outfit(outfit_id)
                st.rerun()

with right_col:
    # Sticky header
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    st.header("Outfit Details")
    st.markdown('</div>', unsafe_allow_html=True)

    # Display selected outfit details if any
    if st.session_state.selected_outfit is not None:
        outfit_id = st.session_state.selected_outfit
        outfit = st.session_state.outfits[outfit_id]

        st.subheader(outfit['name'])

        # Display the interaction status if any
        interaction = st.session_state.outfit_interactions.get(outfit_id, None)
        if interaction:
            status_color = "green" if interaction == "like" else "red"
            status_text = "Liked" if interaction == "like" else "Disliked"
            st.markdown(f"<div style='color:{status_color}; font-weight:bold;'>Status: {status_text}</div>",
                        unsafe_allow_html=True)

        # Display metadata as text
        metadata_text = format_metadata_as_text(outfit['metadata'])
        st.markdown(metadata_text)

        # Add button to clear selection
        if st.button("Close Details"):
            st.session_state.selected_outfit = None
            st.rerun()
    else:
        st.write("Select an outfit to view its details.")

        # Display the update formula for reference
        st.markdown("""
        ### Preference Update Formula

        For each feature in the outfit, the user's preference score is updated using:

        **u<sub>f</sub><sup>new</sup> = u<sub>f</sub><sup>old</sup> + η · W · v<sub>f</sub>**

        Where:
        - u<sub>f</sub><sup>old</sup> = current user preference
        - η = learning rate (adjustable in left panel)
        - W = interaction weight (+1 for like, -1 for dislike)
        - v<sub>f</sub> = feature value in the outfit (0-1)
        """, unsafe_allow_html=True)