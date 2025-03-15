import streamlit as st
import pandas as pd
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import requests
import os
import time

# Set page configuration
st.set_page_config(layout="wide", page_title="Fashion Recommender")

# Constants
# No longer need the hardcoded dataset path since we're using file upload
# DATASET_PATH = "full_data/🔵real_outfits.parquet"  # Removed this line

# Initialize session state
if 'outfits_loaded' not in st.session_state:
    st.session_state.outfits_loaded = False

if 'outfit_images' not in st.session_state:
    st.session_state.outfit_images = {}

if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}

if 'features' not in st.session_state:
    st.session_state.features = {}

if 'ranked_outfits' not in st.session_state:
    st.session_state.ranked_outfits = []

if 'page_number' not in st.session_state:
    st.session_state.page_number = 1

# Track enabled/disabled categories
if 'enabled_categories' not in st.session_state:
    st.session_state.enabled_categories = {}

# Track expander open/closed states
if 'expander_states' not in st.session_state:
    st.session_state.expander_states = {}


# FUNCTIONS

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


# Lazy loading optimization for preference visualization
@st.cache_data(ttl=1)
def get_user_preference_charts(user_preferences):
    """Pre-render all user preference charts at once"""
    charts = {}
    for category, preferences in user_preferences.items():
        if preferences:
            charts[category] = create_preference_chart(preferences, category, is_user_preferences=True)
    return charts


# Function to load and parse outfit data from Parquet file path
@st.cache_data
def load_outfit_data(file_path):
    """Load outfits from parquet file path and extract feature sets"""
    # Check if file exists
    if not os.path.exists(file_path):
        st.error(f"Dataset file not found at: {file_path}")
        st.info(f"Current working directory: {os.getcwd()}")
        return [], {}

    # For large files, show a loading message
    with st.spinner(f"Loading outfit data from {file_path}... This may take a moment."):
        # Load the parquet file
        df = pd.read_parquet(file_path)
        outfits = []

        # First, preprocess the dataframe columns
        for col in ['color_scores', 'textile_scores', 'pattern_scores', 'context_scores', 'weather_scores']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: json.loads(x.replace("'", "\"")) if isinstance(x, str) else x)

        # Extract all unique features across outfits
        features = {
            'Color': set(),
            'Textile': set(),
            'Visual patterns': set(),
            'Context': set(),
            'Weather': set()
        }

        # Parse each row into our outfit structure
        for i, row in df.iterrows():
            try:
                outfit_metadata = {
                    'Color': row.get('color_scores', {}),
                    'Textile': row.get('textile_scores', {}),
                    'Visual patterns': row.get('pattern_scores', {}),
                    'Context': row.get('context_scores', {}),
                    'Weather': row.get('weather_scores', {})
                }

                # Update the features sets
                for category, values in outfit_metadata.items():
                    if values:  # Check if the dictionary is not empty
                        features[category].update(values.keys())

                outfit = {
                    'id': i,
                    'outfit_id': row.get('Outfit ID', f"ID-{i}"),  # Use the actual UUID from the Outfit ID column
                    'name': f"Outfit {i + 1}",
                    'metadata': outfit_metadata,
                    'image_base64': row.get('image', None),
                    'image_url': row.get('image_link', None)
                }
                outfits.append(outfit)

            except Exception as e:
                st.warning(f"Error parsing row {i}: {e}")
                continue

        # Convert sets to sorted lists
        for category in features:
            features[category] = sorted(list(features[category]))

    return outfits, features


# Load user preferences from JSON
@st.cache_data
def load_user_preferences(file):
    try:
        content = file.read()
        json_data = json.loads(content)
        return json_data.get('user_preferences', {})
    except Exception as e:
        st.error(f"Error loading preferences: {e}")
        return {}


# Create a placeholder image for outfits without images
def create_placeholder_image(outfit_id):
    color_r = (outfit_id * 50) % 256
    color_g = (outfit_id * 30) % 256
    color_b = (outfit_id * 70) % 256
    img = Image.new('RGB', (300, 400), (color_r, color_g, color_b))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial", 40)
    except IOError:
        font = ImageFont.load_default()
    draw.text((100, 180), f"Outfit {outfit_id + 1}", fill=(255, 255, 255), font=font)
    return img


# Get an outfit image (from base64, URL, or placeholder)
def get_outfit_image(outfit, score=None):
    outfit_id = outfit['id']

    if outfit_id not in st.session_state.outfit_images:
        img = None
        # Try base64
        if outfit.get('image_base64') and outfit['image_base64'] != 'nan' and outfit['image_base64'] is not None:
            try:
                base64_str = outfit['image_base64']
                if isinstance(base64_str, str) and ',' in base64_str:
                    base64_str = base64_str.split(',')[1]
                img_data = base64.b64decode(base64_str)
                img = Image.open(BytesIO(img_data))
            except:
                img = None

        # Try URL
        if img is None and outfit.get('image_url') and outfit['image_url'] != 'nan' and outfit['image_url'] is not None:
            try:
                response = requests.get(outfit['image_url'])
                img = Image.open(BytesIO(response.content))
            except:
                img = None

        # Fallback to placeholder
        if img is None:
            img = create_placeholder_image(outfit_id)

        st.session_state.outfit_images[outfit_id] = img
    else:
        img = st.session_state.outfit_images[outfit_id]

    # Add score if provided
    if score is not None:
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)

        # Get dimensions to create a proportionally consistent rectangle
        img_width, img_height = img_copy.size

        # Fixed parameters for the score box - updated dimensions
        rect_width = int(img_width * 0.10)  # 10% of image width
        rect_height = int(img_height * 0.04)  # 4% of image height

        # Ensure minimum size
        rect_width = max(rect_width, 50)
        rect_height = max(rect_height, 20)

        # Create semi-transparent black rectangle in top left corner
        draw.rectangle((10, 10, 10 + rect_width, 10 + rect_height), fill=(0, 0, 0, 180))

        # Calculate font size based on rectangle size
        font_size = int(rect_height * 0.6)  # 60% of rectangle height

        try:
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Draw score in the rectangle
        text_x = 15
        text_y = 10 + (rect_height - font_size) // 2  # Center text vertically
        draw.text((text_x, text_y), f"{score:.2f}", fill=(255, 255, 255), font=font)
        img = img_copy

    # Convert to bytes
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr


# Calculate relevance score and breakdown - MODIFIED to include negative scores
def calculate_relevance_score(outfit, user_preferences):
    total_score = 0
    score_breakdown = {}

    for category, metadata in outfit['metadata'].items():
        # Skip disabled categories
        if category in st.session_state.enabled_categories and not st.session_state.enabled_categories[category]:
            continue

        if category not in user_preferences:
            continue

        category_score = 0
        feature_scores = {}

        for feature, outfit_value in metadata.items():
            if feature not in user_preferences[category]:
                continue

            user_pref = user_preferences[category][feature]
            contribution = user_pref * outfit_value

            # Include both positive and negative contributions
            category_score += contribution
            feature_scores[feature] = contribution

        total_score += category_score

        if feature_scores:  # Include category even if total is negative
            score_breakdown[category] = {
                'total': category_score,
                'features': feature_scores
            }

    return total_score, score_breakdown


# Rank outfits by relevance
def rank_outfits(outfits, user_preferences):
    ranked_outfits = []

    for outfit in outfits:
        relevance_score, score_breakdown = calculate_relevance_score(outfit, user_preferences)
        ranked_outfits.append({
            'outfit': outfit,
            'relevance_score': relevance_score,
            'score_breakdown': score_breakdown
        })

    ranked_outfits.sort(key=lambda x: x['relevance_score'], reverse=True)
    return ranked_outfits


# Format score breakdown as text in the requested format - MODIFIED to show only top-3 contributors
def format_score_breakdown(score_breakdown, relevance_score):
    text = f"Total Relevance: {relevance_score:.2f}\n\n"

    # Sort categories by absolute score value (descending)
    sorted_categories = sorted(score_breakdown.items(),
                               key=lambda x: abs(x[1]['total']),
                               reverse=True)

    for category, data in sorted_categories:
        # Only include categories that have at least one non-zero contributor
        if data['features'] and any(abs(score) > 0 for score in data['features'].values()):
            text += f"{category}: {data['total']:.2f}\n"

            # Sort features by absolute score value (descending)
            sorted_features = sorted(data['features'].items(),
                                     key=lambda x: abs(x[1]),
                                     reverse=True)

            # Take top 3 non-zero contributors only
            count = 0
            for feature, score in sorted_features:
                if abs(score) > 0:  # Only include non-zero contributors
                    text += f"   {feature}: {score:.2f}\n"
                    count += 1
                    if count >= 3:  # Limit to top 3
                        break

            # Add an indicator if there are more contributors not shown
            if len([score for feature, score in sorted_features if abs(score) > 0]) > 3:
                text += "   ...\n"

            text += "\n"

    return text


# Function to get downloadable preferences JSON
def get_downloadable_preferences_json():
    """Converts the current user preferences to a JSON string for download"""
    preferences_data = {
        "user_preferences": st.session_state.user_preferences,
        "enabled_categories": st.session_state.enabled_categories,  # Also save enabled/disabled status
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    json_str = json.dumps(preferences_data, indent=4)
    return json_str


# Apply preferences and rerank outfits
def apply_preferences():
    # Remember which expanders were open before reranking
    # (The entire page will be rerun after this operation)
    expander_states = st.session_state.expander_states.copy()

    with st.spinner("Reranking outfits based on preferences..."):
        # Rank outfits with current preferences
        ranked_outfits = rank_outfits(st.session_state.outfits, st.session_state.user_preferences)

        # Pre-compute explanations
        for item in ranked_outfits:
            item['explanation'] = format_score_breakdown(
                item['score_breakdown'],
                item['relevance_score']
            )

        st.session_state.ranked_outfits = ranked_outfits

        # Restore expander states
        st.session_state.expander_states = expander_states


# New function to reset all preferences to zero
def reset_all_preferences():
    # Reset all user preferences to zero
    for category, feature_dict in st.session_state.user_preferences.items():
        for feature in feature_dict:
            # Update both the session state dictionary and any active slider widgets
            st.session_state.user_preferences[category][feature] = 0.0
            # Also update the slider widget state directly
            slider_key = f"slider_{category}_{feature}"
            if slider_key in st.session_state:
                st.session_state[slider_key] = 0.0

    # Re-apply the reset preferences to update the rankings
    apply_preferences()

    # Go back to first page
    st.session_state.page_number = 1


# Functions for pagination
def go_to_next_page():
    total_outfits = len(st.session_state.ranked_outfits)
    outfits_per_page = 100  # Changed to 100 outfits per page
    total_pages = (total_outfits + outfits_per_page - 1) // outfits_per_page

    if st.session_state.page_number < total_pages:
        st.session_state.page_number += 1


def go_to_prev_page():
    if st.session_state.page_number > 1:
        st.session_state.page_number -= 1


def go_to_first_page():
    st.session_state.page_number = 1


def go_to_last_page():
    total_outfits = len(st.session_state.ranked_outfits)
    outfits_per_page = 100  # Changed to 100 outfits per page
    total_pages = (total_outfits + outfits_per_page - 1) // outfits_per_page
    st.session_state.page_number = total_pages


# MAIN APP LOGIC

# Main Interface with file upload option
if not st.session_state.outfits_loaded:
    st.title("Fashion Recommender - Manual Preference Control")
    st.markdown("### Upload Your Outfit Dataset")

    # File uploader for the dataset
    uploaded_file = st.file_uploader("Upload a parquet file containing outfit data", type=['parquet'])

    if uploaded_file is not None:
        try:
            # Load data from uploaded file
            with st.spinner("Loading outfit data from uploaded file... This may take a moment."):
                # Save the uploaded file to a temporary location
                temp_path = "temp_outfits.parquet"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Use the load_outfit_data function with the temporary path
                outfits, features = load_outfit_data(temp_path)

                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            if not outfits:
                st.error("No valid outfits found in the file.")
                st.stop()

            # Store in session state
            st.session_state.outfits = outfits
            st.session_state.features = features
            st.session_state.outfits_loaded = True

            # Initialize empty user preferences for all features
            user_preferences = {}
            for category, feature_list in features.items():
                user_preferences[category] = {feature: 0.0 for feature in feature_list}

            st.session_state.user_preferences = user_preferences

            # Initialize all categories as enabled
            st.session_state.enabled_categories = {category: True for category in features.keys()}

            # Initial ranking with neutral preferences
            with st.spinner("Initial ranking of outfits..."):
                ranked_outfits = rank_outfits(outfits, user_preferences)

                # Pre-compute explanations
                for item in ranked_outfits:
                    item['explanation'] = format_score_breakdown(
                        item['score_breakdown'],
                        item['relevance_score']
                    )

                st.session_state.ranked_outfits = ranked_outfits

            # Success message
            st.success(f"Dataset loaded successfully with {len(outfits)} outfits!")
            st.info("The app will now refresh to show the outfit recommendations...")
            st.rerun()  # Updated from experimental_rerun to rerun

        except Exception as e:
            st.error(f"Error loading outfit data: {e}")
            st.exception(e)  # Show detailed error
            st.stop()
    else:
        # Provide instructions for the file format
        st.info("""
        Please upload a parquet file containing outfit data. The file should have columns:
        - 'color_scores', 'textile_scores', 'pattern_scores', 'context_scores', 'weather_scores' with JSON-formatted feature scores
        - 'image' or 'image_link' for outfit visuals
        - 'Outfit ID' for unique outfit identifiers
        """)
        st.stop()

# Skip to main UI if data is loaded
if not st.session_state.outfits_loaded:
    st.stop()

# Main Interface with manual preference controls and outfits
st.title("Fashion Recommender - Manual Preference Control")

# Create a two-column layout for preferences and outfits
left_col, right_col = st.columns([1, 3])

# Left column with preference controls
with left_col:
    st.header("User Preferences")

    # Top button row with Apply and Reset buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 APPLY", key="apply_prefs_top", use_container_width=True, type="primary"):
            apply_preferences()
    with col2:
        if st.button("🔃 RESET ALL", key="reset_prefs", use_container_width=True, type="secondary"):
            # Call reset function that will set all sliders to zero
            reset_all_preferences()

    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)

    # Option to upload preferences from JSON
    with st.expander("Upload Preferences from JSON", expanded=False):
        preferences_file = st.file_uploader("Upload JSON file", type=['json'])

        if preferences_file:
            try:
                loaded_preferences = load_user_preferences(preferences_file)

                if loaded_preferences:
                    # Merge with existing features to ensure we have all needed features
                    for category, features_dict in loaded_preferences.items():
                        if category in st.session_state.user_preferences:
                            # Update only existing features from the loaded preferences
                            for feature, value in features_dict.items():
                                if feature in st.session_state.user_preferences[category]:
                                    st.session_state.user_preferences[category][feature] = value

                    # Try to load enabled/disabled categories if present
                    try:
                        content = preferences_file.read()
                        json_data = json.loads(content)
                        if "enabled_categories" in json_data:
                            st.session_state.enabled_categories = json_data["enabled_categories"]
                    except:
                        pass  # Keep current enabled/disabled state if not in file

                    st.success("Preferences loaded successfully!")
                    # Apply preferences immediately after loading
                    apply_preferences()
                else:
                    st.error("No valid preferences found in the file.")
            except Exception as e:
                st.error(f"Error loading preferences: {e}")

    # Display search/filter for features
    search_term = st.text_input("🔍 Search features", "")

    # Manual preference controls using sliders, now in collapsible sections
    for category, features in st.session_state.features.items():
        if not features:  # Skip empty categories
            continue

        # Skip categories with no matching search term
        if search_term and not any(search_term.lower() in feature.lower() for feature in features):
            continue

        # Add a header for the category
        st.markdown(f"### {category}")

        # Add enable/disable toggle between header and expander
        enabled = st.session_state.enabled_categories.get(category, True)
        st.session_state.enabled_categories[category] = st.checkbox(
            f"Include {category} in ranking",
            value=enabled,
            key=f"enable_{category}",
            help=f"When unchecked, {category} features will be excluded from outfit ranking"
        )

        # Check if we have a saved state for this expander
        if category not in st.session_state.expander_states:
            st.session_state.expander_states[category] = False

        # Create expander for the category, using the saved state but with a specific label for each category
        with st.expander(f"{category} preferences", expanded=st.session_state.expander_states[category]):
            # Save the expander state when it changes
            st.session_state.expander_states[category] = True

            # Filter features by search term
            if search_term:
                filtered_features = [f for f in features if search_term.lower() in f.lower()]
            else:
                filtered_features = features

            # Show all features in the expanded section
            for feature in filtered_features:
                current_value = st.session_state.user_preferences[category].get(feature, 0.0)
                st.session_state.user_preferences[category][feature] = st.slider(
                    f"{feature}",
                    min_value=-1.0,
                    max_value=1.0,
                    value=current_value,
                    step=0.1,
                    key=f"slider_{category}_{feature}"
                )

            # If there are many features, show a message instead of another expander
            if len(filtered_features) > 10 and not search_term:
                st.markdown(f"Showing all {len(filtered_features)} features. Use the search box to filter them.")

    # Option to download current preferences
    if st.button("Download Current Preferences"):
        json_str = get_downloadable_preferences_json()
        st.download_button(
            label="Click to Download",
            data=json_str,
            file_name="fashion_preferences.json",
            mime="application/json"
        )

    # Preference charts removed as requested

# Right column with ranked outfits
with right_col:
    st.header("Ranked Outfits")

    # Pagination for large outfit sets (using buttons)
    total_outfits = len(st.session_state.ranked_outfits)
    outfits_per_page = 100  # Changed to 100 outfits per page
    total_pages = (total_outfits + outfits_per_page - 1) // outfits_per_page

    # Calculate start and end indices for the current page
    start_idx = (st.session_state.page_number - 1) * outfits_per_page
    end_idx = min(start_idx + outfits_per_page, total_outfits)

    # Create button layout for pagination
    if total_pages > 1:
        # First show the outfit count
        st.write(f"Showing outfits {start_idx + 1} to {end_idx} of {total_outfits}")

        # Create a layout with buttons filling their containers
        # Use 5 columns with appropriate proportions
        # Left (First+Prev) | Center (Page) | Right (Next+Last)
        btn_col1, btn_col2, page_col, btn_col3, btn_col4 = st.columns([1, 1, 3, 1, 1])

        with btn_col1:
            if st.button("⏮️ First", disabled=(st.session_state.page_number == 1), key="first_btn",
                         use_container_width=True):
                go_to_first_page()

        with btn_col2:
            if st.button("◀️ Prev", disabled=(st.session_state.page_number == 1), key="prev_btn",
                         use_container_width=True):
                go_to_prev_page()

        with page_col:
            # Center align the page text with smaller font size
            st.markdown(
                f"<p style='text-align: center; font-size: 0.9rem;'>Page {st.session_state.page_number}/{total_pages}</p>",
                unsafe_allow_html=True)

        with btn_col3:
            if st.button("Next ▶️", disabled=(st.session_state.page_number == total_pages), key="next_btn",
                         use_container_width=True):
                go_to_next_page()

        with btn_col4:
            if st.button("Last ⏭️", disabled=(st.session_state.page_number == total_pages), key="last_btn",
                         use_container_width=True):
                go_to_last_page()

    # Display outfits in a grid with explanations directly underneath
    paged_outfits = st.session_state.ranked_outfits[start_idx:end_idx]
    cols_per_row = 4  # Adjusted to 4 columns for a better fit with 100 outfits
    rows = (len(paged_outfits) + cols_per_row - 1) // cols_per_row  # Ceiling division

    # Create container for each row
    for row in range(rows):
        # Create columns within the row
        cols = st.columns(cols_per_row)

        # Fill each column with an outfit and its explanation
        for col in range(cols_per_row):
            idx = row * cols_per_row + col

            # Skip if we've run out of outfits
            if idx >= len(paged_outfits):
                continue

            # Get outfit data
            ranked_item = paged_outfits[idx]
            outfit = ranked_item['outfit']
            score = ranked_item['relevance_score']
            explanation = ranked_item['explanation']

            # Display in this column
            with cols[col]:
                # Show outfit image with score overlay only
                img = get_outfit_image(outfit, score)

                # Display the image first with a simpler caption
                st.image(img, caption=f"Outfit #{idx + 1}", use_container_width=True)

                # Display the outfit ID using st.code() just like the recommendation explanation
                complete_id = outfit['outfit_id']
                st.markdown("<b>Outfit ID:</b>", unsafe_allow_html=True)
                st.code(complete_id)

                # Add a collapsed expander with all outfit metadata
                with st.expander("Show outfit metadata", expanded=False):
                    has_data = False
                    for category, metadata in outfit['metadata'].items():
                        # Filter for non-zero values only
                        non_zero_items = {feature: value for feature, value in metadata.items() if value > 0}

                        if non_zero_items:  # Only show categories with non-zero data
                            has_data = True
                            st.markdown(f"**{category}:**")
                            metadata_text = "\n".join(
                                [f"{feature}: {value:.2f}" for feature, value in non_zero_items.items()])
                            st.code(metadata_text)

                    if not has_data:
                        st.write("No non-zero metadata found for this outfit.")

                # Only display the explanation breakdown without the total score
                st.markdown("<b>Recommendation Breakdown:</b>", unsafe_allow_html=True)

                # Extract only category breakdowns (skip the total score lines)
                lines = explanation.split('\n')
                category_start = 2  # Skip the "Total Relevance: X.XX" and the blank line after it
                detailed_explanation = '\n'.join(lines[category_start:])

                st.code(detailed_explanation)

                # Add a small gap
                st.write("")