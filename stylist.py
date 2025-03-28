import streamlit as st
import pandas as pd
import numpy as np
import json
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from PIL import Image


@st.cache_resource
def load_torch():
    """Lazy load torch to improve startup time"""
    try:
        import torch
        return torch
    except ImportError:
        st.error("Could not import torch. Please install it with: pip install torch")
        return None


@st.cache_resource
def load_fashion_models():
    """Lazy load the fashion models and transformers to improve startup time"""
    try:
        # Import necessary modules
        import open_clip
        from sentence_transformers.util import semantic_search

        # Import torch using our lazy loader
        torch = load_torch()
        if torch is None:
            return None, None, None, None, None

        # Load the model, transformers, and tokenizer
        fashion_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:Marqo/marqo-fashionSigLIP')
        fashion_tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionSigLIP')

        return fashion_model, preprocess_train, preprocess_val, fashion_tokenizer, semantic_search
    except ImportError as e:
        st.error(f"Error loading fashion models: {e}")
        st.error("Please install required packages: pip install open-clip-torch sentence-transformers")
        return None, None, None, None, None


# Replace your existing similarity_search function with this version that uses lazy loading
def similarity_search(query, df, id_column='id', embeds_column='embeds', top_k=5, device=None):
    """
    Perform semantic search using FashionSigLIP model and return the top_k rows of the filtered DataFrame,
    sorted by relevance score (highest first).
    """
    if embeds_column not in df.columns:
        raise ValueError(f"Column '{embeds_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")

    # Lazy load torch
    torch = load_torch()
    if torch is None:
        st.error("Could not load torch. Using fallback similarity search.")
        return df.head(top_k)

    # Set device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Lazy load the model and related functions
    fashion_model, _, preprocess_val, fashion_tokenizer, semantic_search_func = load_fashion_models()

    if fashion_model is None:
        st.error("Could not load fashion models. Using fallback similarity search.")
        return df.head(top_k)

    fashion_model.to(device)
    fashion_model.eval()

    # Case 1: Handle integer as DataFrame index
    if isinstance(query, int) and 0 <= query < len(df):
        ref_row = df.iloc[query]
        ref_id = ref_row[id_column]
        query_embeds = np.array(ref_row[embeds_column], dtype=np.float32)
        # Exclude the reference row
        filtered_df = df[df[id_column] != ref_id].copy()
        query_type = "Index"

    # Case 2: Handle string as ID lookup or text query
    elif isinstance(query, str):
        # Check if it's an ID lookup or a text query
        ref_row = df[df[id_column] == query]

        if not ref_row.empty:
            # It's an ID lookup
            ref_row = ref_row.iloc[0]
            query_embeds = np.array(ref_row[embeds_column], dtype=np.float32)
            # Exclude the reference row
            filtered_df = df[df[id_column] != query].copy()
            query_type = "ID"
        else:
            # It's a text query
            text_inputs = fashion_tokenizer([query]).to(device)
            with torch.no_grad():
                query_embeds = fashion_model.encode_text(text_inputs).cpu().numpy()[0]
            filtered_df = df.copy()
            query_type = "Text"

    # Case 3: Handle image query
    elif isinstance(query, Image.Image):
        img_tensor = preprocess_val(query).unsqueeze(0).to(device)
        with torch.no_grad():
            query_embeds = fashion_model.encode_image(img_tensor).cpu().numpy()[0]
        filtered_df = df.copy()
        query_type = "Image"

    else:
        raise ValueError("Query must be either a text string, an image, or an integer index/ID.")

    # Convert to tensor (optimized)
    query_tensor = torch.tensor(np.array([query_embeds], dtype=np.float32), dtype=torch.float32, device=device)

    # Convert list of embeddings to numpy array first (prevents the slow list-to-tensor conversion)
    corpus_embeds = np.vstack(filtered_df[embeds_column].values).astype(np.float32)
    corpus_tensor = torch.tensor(corpus_embeds, dtype=torch.float32, device=device)

    # Perform semantic search using cosine similarity
    hits = semantic_search_func(query_tensor, corpus_tensor, top_k=top_k)[0]

    # Retrieve top results and store their indices and scores
    top_indices = [hit['corpus_id'] for hit in hits]
    scores = [hit['score'] for hit in hits]

    # Filter the DataFrame to return only the top_k rows
    result_df = filtered_df.iloc[top_indices].reset_index(drop=True)

    # Add the scores as a new column in the filtered DataFrame
    result_df['score'] = scores

    # Add query type for reference
    result_df['query_type'] = query_type

    # Sort the filtered DataFrame by the relevance score (highest first)
    result_df = result_df.sort_values(by='score', ascending=False).reset_index(drop=True)

    return result_df


# OpenAI API key - using the original key as in the initial code
api_key = st.secrets["openai"]["api_key"]
client = OpenAI(api_key=api_key)


class OutfitCandidate(BaseModel):
    description: str
    items: List[str]


class OutfitCandidates(BaseModel):
    candidate_1: OutfitCandidate
    candidate_2: OutfitCandidate
    candidate_3: OutfitCandidate
    candidate_4: OutfitCandidate
    candidate_5: OutfitCandidate


stylist_prompt = """ 
You are a professional stylist who provides premium styling service. You will be provided with information about your client and their preferences. 

Your goal is to create a text description of 5 outfits, and also decompose them into a list of specific items.  Outfits should be relevant to the user context but diverse enough. 

When composing outfits, consider the user's color type. 
Here is the definition of each color type and rules for styling: 

SPRING color type
- Peachy skin, golden blonde/red hair, light eyes.
✅ Do:
- Light warm colors (peach, coral, aqua, mint)
- Cream/ivory instead of white
- Soft, light fabrics (silk, cotton)
- Playful, fresh combinations

❌ Avoid Harsh blacks, deep navy. Cool greys or icy tonesHeavy or muted fabrics/colors


SUMMER color type
Ashy hair, pink undertones, light eyes.

✅ Do:
- Muted cool tones (lavender, dusty rose, soft blue)
- Light grey, rose beige
- Matte, flowy fabrics (linen, suede)
- Tonal or low-contrast outfits

❌ Avoid: 
- Warm earthy tones (camel, rust)
- High-contrast combos
- Shiny, harsh textures


AUTUMN color type
Rich hair (red/brown), warm skin, hazel/green eyes.

✅ Do:
- Earthy tones (olive, rust, mustard, brown)
- Deep teal, forest green
- Textured fabrics (wool, tweed, suede)
- Layered, cozy looks

❌ AVOID: 
- Icy pastels, neon brights
- Cool tones (blue-grey, fuchsia)
- Pure black or white


WINTER color type
Dark hair, pale skin, bright/cool eyes.

✅ Do:
- Bold, cool tones (black, white, red, emerald, cobalt)
- Icy pastels (icy blue, icy pink)
- Sleek, structured fabrics (leather, crisp cotton)
- Strong contrast in outfits

❌ AVOID: 
- Warm earthy tones (beige, orange, camel)
- Washed-out colors
- Soft, muddy blends

Return only a json in the following format: 
{
"candidate_1": {
    "description": "A sophisticated, professional outfit that is both comfortable and stylish. Perfect for a full day of work and meetings, this ensemble exudes confidence.",
    "reason":"matches perfectly to users color type and lifestyle", 
    "items": [
      "A tailored navy blue blazer with a slim fit",
      "A white silk blouse with a slight v-neck for elegance",
      "Black slim-fit trousers that elongate the legs",
      "A classic black leather tote bag for practicality and style",
      "Black pointed-toe pumps for a sleek, polished look",
      "A delicate silver bracelet",
      "stud earrings for a touch of sophistication"
    ]
  }, 
"candidate_2": {...}, 
"candidate_3": {...}
}

Make sure that each item description refers to only one item to ensure good semantic retrieval quality
"""


@st.cache_data
def user_description_to_outfits(prompt, user_description, api_key=api_key, response_format=OutfitCandidates):
    """Cache the LLM output to avoid unnecessary API calls"""
    client = OpenAI(api_key=api_key)

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_description}
            ],
            max_tokens=1500,
            temperature=0.1,
            response_format=response_format
        )

        response_content = completion.choices[0].message.parsed.model_dump()
        return json.dumps(response_content)

    except Exception as e:
        st.error(f"Error generating outfit candidates: {e}")
        # Return a basic structure to prevent further errors
        return json.dumps({
            "candidate_1": {"description": "Error generating outfit", "items": ["No items available"]},
            "candidate_2": {"description": "Error generating outfit", "items": ["No items available"]},
            "candidate_3": {"description": "Error generating outfit", "items": ["No items available"]},
            "candidate_4": {"description": "Error generating outfit", "items": ["No items available"]},
            "candidate_5": {"description": "Error generating outfit", "items": ["No items available"]}
        })


# Set page config with minimal initial loading
st.set_page_config(page_title="Stylist Helper Demo", layout="wide")

# Initialize session state for storing uploaded data
if 'items_df' not in st.session_state:
    st.session_state.items_df = None
if 'outfits_df' not in st.session_state:
    st.session_state.outfits_df = None
if 'session_outfits_df' not in st.session_state:
    st.session_state.session_outfits_df = None
if 'datasets_loaded' not in st.session_state:
    st.session_state.datasets_loaded = False
if 'items_parts_loaded' not in st.session_state:
    st.session_state.items_parts_loaded = []
if 'outfits_parts_loaded' not in st.session_state:
    st.session_state.outfits_parts_loaded = []
if 'total_items_count' not in st.session_state:
    st.session_state.total_items_count = 0
if 'total_outfits_count' not in st.session_state:
    st.session_state.total_outfits_count = 0
if 'items_filenames' not in st.session_state:
    st.session_state.items_filenames = []
if 'outfits_filenames' not in st.session_state:
    st.session_state.outfits_filenames = []
if 'current_items_file' not in st.session_state:
    st.session_state.current_items_file = None
if 'current_outfits_file' not in st.session_state:
    st.session_state.current_outfits_file = None
if 'reference_images' not in st.session_state:
    st.session_state.reference_images = []
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False


# Process items dataset parts
def process_items_parts():
    if not st.session_state.items_parts_loaded:
        return False

    try:
        # Extract dataframes from parts
        part_dfs = [part["dataframe"] for part in st.session_state.items_parts_loaded]

        # Combine all parts
        items_df = pd.concat(part_dfs, ignore_index=True)

        # Remove duplicates if any
        items_df = items_df.drop_duplicates(subset=['item_id'])
        st.session_state.items_df = items_df

        return True
    except Exception as e:
        st.error(f"Error combining items parts: {e}")
        return False


# Process outfits dataset parts
def process_outfits_parts():
    if not st.session_state.outfits_parts_loaded:
        return False

    try:
        # Extract dataframes from parts
        part_dfs = [part["dataframe"] for part in st.session_state.outfits_parts_loaded]

        # Combine all parts
        outfits_df = pd.concat(part_dfs, ignore_index=True)

        # Remove duplicates if any
        outfits_df = outfits_df.drop_duplicates(subset=['id'])
        st.session_state.outfits_df = outfits_df

        # Filter for session outfits
        session_outfits_df = outfits_df[outfits_df['type'] == 'SESSION']
        st.session_state.session_outfits_df = session_outfits_df

        return True
    except Exception as e:
        st.error(f"Error combining outfits parts: {e}")
        return False


# Check if conditions are met to show "Start Stylist" button
def can_proceed_to_stylist():
    return (st.session_state.items_df is not None and
            st.session_state.outfits_df is not None and
            st.session_state.session_outfits_df is not None)


# Process items file
def handle_items_upload(items_file):
    if items_file is None:
        return

    # Skip if this is the same file as before
    if st.session_state.current_items_file == items_file:
        return

    # Store current file reference
    st.session_state.current_items_file = items_file

    try:
        with st.spinner("Processing items dataset..."):
            items_part = pd.read_parquet(items_file)
            # Filter out rows without embeddings
            items_part = items_part[items_part['embeds'].notna()]

            # Add to parts list (only if not already added)
            if items_file.name not in st.session_state.items_filenames:
                file_info = {
                    "filename": items_file.name,
                    "count": len(items_part),
                    "dataframe": items_part
                }

                st.session_state.items_parts_loaded.append(file_info)
                st.session_state.items_filenames.append(items_file.name)
                st.session_state.total_items_count += len(items_part)

                # Automatically process the parts
                process_items_parts()
            else:
                st.warning(f"File {items_file.name} was already added")

    except Exception as e:
        st.error(f"Error processing items file: {e}")


# Process outfits file
def handle_outfits_upload(outfits_file):
    if outfits_file is None:
        return

    # Skip if this is the same file as before
    if st.session_state.current_outfits_file == outfits_file:
        return

    # Store current file reference
    st.session_state.current_outfits_file = outfits_file

    try:
        with st.spinner("Processing outfits dataset..."):
            outfits_part = pd.read_parquet(outfits_file)
            # Filter out rows without embeddings
            outfits_part = outfits_part[outfits_part['embeds'].notna()]

            # Add to parts list (only if not already added)
            if outfits_file.name not in st.session_state.outfits_filenames:
                file_info = {
                    "filename": outfits_file.name,
                    "count": len(outfits_part),
                    "dataframe": outfits_part
                }

                st.session_state.outfits_parts_loaded.append(file_info)
                st.session_state.outfits_filenames.append(outfits_file.name)
                st.session_state.total_outfits_count += len(outfits_part)

                # Automatically process the parts
                process_outfits_parts()
            else:
                st.warning(f"File {outfits_file.name} was already added")

    except Exception as e:
        st.error(f"Error processing outfits file: {e}")


# Process reference images
def handle_reference_images(uploaded_files):
    if not uploaded_files:
        st.session_state.reference_images = []
        return

    # Check if reference images have changed
    current_count = len(st.session_state.reference_images) if hasattr(st.session_state, 'reference_images') else 0
    new_count = len(uploaded_files)

    # If count changed, we need to rerun analysis
    if current_count != new_count:
        st.session_state.results_ready = False

    try:
        images = []
        # Store both the PIL images and their file names
        for i, file in enumerate(uploaded_files):
            image = Image.open(file)
            file_name = file.name if hasattr(file, 'name') else f"Image {i + 1}"
            images.append({"image": image, "name": file_name})

        st.session_state.reference_images = images
    except Exception as e:
        st.error(f"Error processing reference images: {e}")


# Always display the header - regardless of mode
# Create a header row with title and START STYLIST button on the right
title_col, button_col = st.columns([3, 1])

with title_col:
    st.title("Stylist Helper")

# Show START STYLIST button on the right side when conditions are met
with button_col:
    if not st.session_state.datasets_loaded and can_proceed_to_stylist():
        if st.button("START STYLIST", type="primary", use_container_width=True):
            st.session_state.datasets_loaded = True
            st.rerun()

# File upload section in the main part of the screen if datasets aren't loaded yet
if not st.session_state.datasets_loaded:
    # Two column layout for items and outfits
    col1, col2 = st.columns(2)

    # Left column - Items
    with col1:
        st.subheader("Items Dataset")

        # Always display status line to maintain layout consistency
        items_parts = len(st.session_state.items_parts_loaded)
        items_count = st.session_state.total_items_count
        if items_parts > 0:
            st.caption(f"Loaded: {items_parts} parts • {items_count} items")
        else:
            st.caption("Loaded: 0 parts • 0 items")

        # Items uploader - automatically processes on upload
        items_file = st.file_uploader("", type=["parquet"], key="items_uploader", label_visibility="collapsed",
                                      on_change=lambda: handle_items_upload(st.session_state.items_uploader))

    # Right column - Outfits
    with col2:
        st.subheader("Outfits Dataset")

        # Always display status line to maintain layout consistency
        outfits_parts = len(st.session_state.outfits_parts_loaded)
        outfits_count = st.session_state.total_outfits_count
        if outfits_parts > 0:
            st.caption(f"Loaded: {outfits_parts} parts • {outfits_count} outfits")
        else:
            st.caption("Loaded: 0 parts • 0 outfits")

        # Outfits uploader - automatically processes on upload
        outfits_file = st.file_uploader("", type=["parquet"], key="outfits_uploader", label_visibility="collapsed",
                                        on_change=lambda: handle_outfits_upload(st.session_state.outfits_uploader))

    # Hidden documentation to reduce clutter
    with st.expander("Help"):
        st.markdown("""
        ### Upload multiple dataset parts
        1. Upload items and outfits datasets
        2. Press "Run Stylist Helper"
        3. Modify user description 
        4. Press "Create suggestions"

        ### Dataset requirements
        - Items: Must have columns `item_id`, `item_name`, `item_image_link`, `embeds`
        - Outfits: Must have columns `id`, `type`, `name`, `imageLink`, `embeds`
        """)

# Show the stylist interface once datasets are loaded
else:

    # Sidebar for user input
    with st.sidebar:

        # CHANGE 1: Move "Get suggestions" button to the top of the sidebar
        submit_button = st.button("Get suggestions", type="primary", use_container_width=True)

        user_description = st.text_area(
            "Describe the user and their preferences:",
            "Executive-level business woman in her 30s, lives in Austin, Texas. Looks for a ageless look for her work to look professional and noticeable. Open for experiments.",
            height=150
        )

        # Fixed number of relevant outfits to display (21)
        num_outfits = 21

        # New feature: Reference image upload
        ref_images = st.file_uploader(
            "Upload reference images for style inspiration:",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="reference_image_uploader",
            help="Upload images of styles you like for more personalized recommendations"
        )

        # Process uploaded reference images
        if ref_images:
            handle_reference_images(ref_images)

            # Display thumbnails of uploaded reference images
            st.caption(f"{len(ref_images)} reference images uploaded")
            thumb_cols = st.columns(min(3, len(ref_images)))
            for i, img_file in enumerate(ref_images[:3]):  # Show up to 3 thumbnails
                with thumb_cols[i]:
                    img = Image.open(img_file)
                    st.image(img, width=80)

        # REMOVED: Generate button was moved to the top

    # Reset datasets button (small and at the bottom of the sidebar)
    with st.sidebar:
        st.markdown("---")
        if st.button("Reset datasets", key="reset_datasets"):
            st.session_state.items_df = None
            st.session_state.outfits_df = None
            st.session_state.session_outfits_df = None
            st.session_state.datasets_loaded = False
            st.session_state.items_parts_loaded = []
            st.session_state.outfits_parts_loaded = []
            st.session_state.items_filenames = []
            st.session_state.outfits_filenames = []
            st.session_state.total_items_count = 0
            st.session_state.total_outfits_count = 0
            st.session_state.current_items_file = None
            st.session_state.current_outfits_file = None
            st.session_state.reference_images = []
            st.session_state.results_ready = False
            st.rerun()


    # Function to retrieve outfit items - defined only when needed
    def retrieve_outfit(query_strings, items_df):
        results = []
        # Import torch only when needed
        import torch

        for item in query_strings:
            try:
                result = similarity_search(
                    item,
                    items_df[items_df['item_status'] == 'ACTIVE'],
                    id_column='item_id',
                    embeds_column='embeds',
                    top_k=1,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                results.append(result)
            except Exception as e:
                st.error(f"Error retrieving outfit item '{item}': {e}")
                # Create an empty dataframe with the same structure
                empty_df = pd.DataFrame(columns=items_df.columns)
                results.append(empty_df)

        if results:
            try:
                df = pd.concat(results, ignore_index=False)
                return df
            except Exception as e:
                st.error(f"Error concatenating results: {e}")
                return pd.DataFrame(columns=items_df.columns)
        else:
            return pd.DataFrame(columns=items_df.columns)


    # Function to retrieve outfits based on user description
    def retrieve_outfits_by_description(user_description, outfits_df, top_k=15):
        # Import torch only when needed
        import torch

        try:
            results = similarity_search(
                user_description,
                outfits_df,
                id_column='id',
                embeds_column='embeds',
                top_k=top_k,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            return results
        except Exception as e:
            st.error(f"Error retrieving outfits: {e}")
            return pd.DataFrame(columns=outfits_df.columns)


    # Function to retrieve items based on reference images
    def retrieve_items_by_image(reference_images, items_df, top_k=9):  # CHANGE 2: Changed top_k from 3 to 9
        # Import torch only when needed
        import torch

        # Store results with image references
        image_results = []

        for img_data in reference_images:
            try:
                img = img_data["image"]
                img_name = img_data["name"]

                results = similarity_search(
                    img,  # Pass PIL image directly
                    items_df[items_df['item_status'] == 'ACTIVE'],
                    id_column='item_id',
                    embeds_column='embeds',
                    top_k=top_k,  # Now using 9 instead of 3
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )

                # Store the results with reference to the image name and the image itself
                image_results.append({
                    "image_name": img_name,
                    "image": img,
                    "results": results
                })
            except Exception as e:
                st.error(f"Error retrieving items by image: {e}")

        # Return the structured data with image references
        return image_results


    # Function to display initial search results
    def display_initial_search_results(initial_results, num_outfits=6):
        if initial_results.empty:
            st.warning("No matching outfits found.")
            return

        # Limit the number of outfits to display based on user preference
        limited_results = initial_results.head(num_outfits)

        # Create a grid to display outfits
        cols = st.columns(3)

        for i, (_, outfit) in enumerate(limited_results.iterrows()):
            with cols[i % 3]:
                # Display outfit image using the same styling as items below
                if 'imageLink' in outfit and outfit['imageLink']:
                    st.image(
                        outfit['imageLink'],
                        caption=f"{outfit.get('name', 'Outfit')}",
                        width=250  # Same width as item images
                    )
                else:
                    # Placeholder if image not available
                    st.info("Image not available")

                # Create clickable link with outfit ID
                outfit_id = outfit['id']
                outfit_url = f"https://chic-control.com/session-outfit-creation/?id={outfit_id}"
                st.markdown(f"[View Outfit: {outfit_id}]({outfit_url})")

                # Display outfit description if available
                if 'description' in outfit and outfit['description']:
                    st.markdown(f"<div style='font-size: 0.85em;'>{outfit['description']}</div>",
                                unsafe_allow_html=True)

                # Display additional outfit metadata if available
                if 'style' in outfit and outfit['style']:
                    st.markdown(f"**Style:** {outfit['style']}")
                if 'occasion' in outfit and outfit['occasion']:
                    st.markdown(f"**Occasion:** {outfit['occasion']}")


    # Function to display reference-based items
    def display_reference_items(image_results):
        if not image_results:
            return

        # For each reference image and its results
        for image_data in image_results:
            image_name = image_data["image_name"]
            ref_image = image_data["image"]
            results = image_data["results"]

            if results.empty:
                continue

            # Create a 2-column layout for the reference image and heading
            ref_col, heading_col = st.columns([1, 3])

            # Display the reference image in the left column
            with ref_col:
                st.image(ref_image, width=150, caption="Reference")

            # Display the heading in the right column
            with heading_col:
                st.subheader(image_name)

            # Add some space after the heading
            st.write("")

            # Create a grid to display items for this reference image
            cols = st.columns(3)

            for i, (_, item) in enumerate(results.iterrows()):
                with cols[i % 3]:
                    # Display item image
                    if 'item_image_link' in item and item['item_image_link']:
                        st.image(
                            item['item_image_link'],
                            caption=f"{item.get('item_name', 'Item')}",
                            width=250
                        )
                    else:
                        st.info("Image not available")

                    # Create clickable link with item ID
                    item_id = item['item_id']
                    item_url = f"https://chic-control.com/outfit-item/?id={item_id}"
                    st.markdown(f"[View Item: {item_id}]({item_url})")

                    # Convert price from cents to dollars
                    price = item.get('item_price', 'N/A')
                    if isinstance(price, (int, float)):
                        price = f"${price / 100:.2f}"
                    else:
                        price = "N/A"
                    st.markdown(f"**Price:** {price}")
                    st.markdown(f"**Brand:** {item.get('item_shop', 'N/A')}")

                    # Add item description
                    if 'item_description' in item and item['item_description']:
                        st.markdown(f"<div style='font-size: 0.85em;'>{item['item_description']}</div>",
                                    unsafe_allow_html=True)

            # Add separation between different reference image results
            st.markdown("---")


    # Function to display outfit candidates in a grid
    def display_outfit_candidates(candidates, items_df):
        for outfit_num, (candidate, values) in enumerate(candidates.items(), 1):
            st.subheader(f"Outfit {outfit_num}")

            # Reduce font size of outfit description by using markdown
            st.markdown(f"<div style='font-size: 0.9em;'>{values['description']}</div>", unsafe_allow_html=True)

            # Add vertical spacing between description and images
            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
            # Alternatively, you can use multiple empty writes for spacing:
            # st.write("")
            # st.write("")

            outfit_items = values['items']
            df = retrieve_outfit(outfit_items, items_df)

            if df.empty:
                st.warning("No matching items found for this outfit.")
                continue

            # Create a grid to display items
            cols = st.columns(3)

            for i, ((_, item), item_description) in enumerate(zip(df.iterrows(), outfit_items)):
                with cols[i % 3]:
                    # Reduce image card size by setting width
                    if 'item_image_link' in item and item['item_image_link']:
                        st.image(
                            item['item_image_link'],
                            caption=f"{item.get('item_name', 'Item')}",
                            width=250  # Reduced size
                        )
                    else:
                        st.info("Image not available")

                    # Display the LLM-generated item description
                    st.markdown(f"<div style='font-size: 0.85em; font-style: italic;'>{item_description}</div>",
                                unsafe_allow_html=True)

                    # Create clickable link with item ID
                    item_id = item['item_id']
                    item_url = f"https://chic-control.com/outfit-item/?id={item_id}"
                    st.markdown(f"[View Item: {item_id}]({item_url})")

                    # Convert price from cents to dollars
                    price = item.get('item_price', 'N/A')
                    if isinstance(price, (int, float)):
                        price = f"${price / 100:.2f}"
                    else:
                        price = "N/A"
                    st.markdown(f"**Price:** {price}")
                    st.markdown(f"**Brand:** {item.get('item_shop', 'N/A')}")

                    # Add item description
                    if 'item_description' in item and item['item_description']:
                        st.markdown(f"<div style='font-size: 0.85em;'>{item['item_description']}</div>",
                                    unsafe_allow_html=True)

            st.markdown("---")


    def retrieve_outfits_by_image(reference_images, outfits_df, top_k=9):
        # Import torch only when needed
        import torch

        # Store results with image references
        image_results = []

        for img_data in reference_images:
            try:
                img = img_data["image"]
                img_name = img_data["name"]

                results = similarity_search(
                    img,  # Pass PIL image directly
                    outfits_df,  # Using outfits dataframe instead of items
                    id_column='id',  # Using the outfit's id column
                    embeds_column='embeds',
                    top_k=top_k,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )

                # Store the results with reference to the image name and the image itself
                image_results.append({
                    "image_name": img_name,
                    "image": img,
                    "results": results
                })
            except Exception as e:
                st.error(f"Error retrieving outfits by image: {e}")

        # Return the structured data with image references
        return image_results


    def display_reference_outfits(image_results):
        if not image_results:
            return

        # For each reference image and its results
        for image_data in image_results:
            image_name = image_data["image_name"]
            ref_image = image_data["image"]
            results = image_data["results"]

            if results.empty:
                continue

            # Create a 2-column layout for the reference image and heading
            ref_col, heading_col = st.columns([1, 3])

            # Display the reference image in the left column
            with ref_col:
                st.image(ref_image, width=150, caption="Reference")

            # Display the heading in the right column
            with heading_col:
                st.subheader(image_name)

            # Add some space after the heading
            st.write("")

            # Create a grid to display outfits for this reference image
            cols = st.columns(3)

            for i, (_, outfit) in enumerate(results.iterrows()):
                with cols[i % 3]:
                    # Display outfit image
                    if 'imageLink' in outfit and outfit['imageLink']:
                        st.image(
                            outfit['imageLink'],
                            caption=f"{outfit.get('name', 'Outfit')}",
                            width=250
                        )
                    else:
                        st.info("Image not available")

                    # Create clickable link with outfit ID
                    outfit_id = outfit['id']
                    outfit_url = f"https://chic-control.com/session-outfit-creation/?id={outfit_id}"
                    st.markdown(f"[View Outfit: {outfit_id}]({outfit_url})")

                    # Display outfit description if available
                    if 'description' in outfit and outfit['description']:
                        st.markdown(f"<div style='font-size: 0.85em;'>{outfit['description']}</div>",
                                    unsafe_allow_html=True)

                    # Display additional outfit metadata if available
                    if 'style' in outfit and outfit['style']:
                        st.markdown(f"**Style:** {outfit['style']}")
                    if 'occasion' in outfit and outfit['occasion']:
                        st.markdown(f"**Occasion:** {outfit['occasion']}")

            # Add separation between different reference image results
            st.markdown("---")


    # Add this to track user description changes
    if 'previous_description' not in st.session_state:
        st.session_state.previous_description = ""

    # In the sidebar section, after the user_description text_area:
    # Check if description has changed
    if st.session_state.previous_description != user_description:
        # Reset results to force rerun
        st.session_state.results_ready = False
        st.session_state.previous_description = user_description


    # In the ai_stylist function, update to ensure it always generates new results
    def ai_stylist(user_description, items_df, session_outfits_df, num_outfits=21, reference_images=None):
        # Always reset results data to ensure fresh generation
        st.session_state.relevant_outfits = None
        st.session_state.ai_outfits = None
        st.session_state.image_results = None

        # Process all data
        with st.spinner("Generating outfit recommendations..."):
            # Get AI-generated outfits (always generate new ones)
            candidates_json = user_description_to_outfits(
                stylist_prompt,
                user_description,
                response_format=OutfitCandidates
            )
            st.session_state.ai_outfits = json.loads(candidates_json)

            # Get relevant existing outfits (always search again)
            initial_results = retrieve_outfits_by_description(
                user_description,
                session_outfits_df,
                top_k=num_outfits
            )
            st.session_state.relevant_outfits = initial_results

            # Get reference image results if available
            if reference_images and len(reference_images) > 0:
                image_results = retrieve_outfits_by_image(reference_images, session_outfits_df)
                st.session_state.image_results = image_results

            # Mark results as ready to display
            st.session_state.results_ready = True

            # Store current description for future comparison
            st.session_state.previous_description = user_description

        # Rerun to display the results with tabs
        st.rerun()


    # Process button click
    if submit_button and not st.session_state.results_ready:
        ai_stylist(
            user_description,
            st.session_state.items_df,
            st.session_state.session_outfits_df,
            num_outfits,
            st.session_state.reference_images
        )

    # Display results in tabs if they're ready
    if st.session_state.results_ready:
        # Create tab container
        tab1, tab2, tab3 = st.tabs(["Relevant Outfits", "Assembled by AI", "By Reference Photos"])

        # Tab 1: Relevant Outfits
        with tab1:
            if st.session_state.relevant_outfits is not None and not st.session_state.relevant_outfits.empty:
                display_initial_search_results(st.session_state.relevant_outfits, num_outfits)
            else:
                st.info("No relevant existing outfits found that match your style preferences.")

        # Tab 2: AI-Generated Outfits
        with tab2:
            if st.session_state.ai_outfits is not None:
                display_outfit_candidates(st.session_state.ai_outfits, st.session_state.items_df)
            else:
                st.info("No AI-generated outfits available.")

        # Tab 3: By Reference Photos
        with tab3:
            if st.session_state.image_results and len(st.session_state.image_results) > 0:
                display_reference_outfits(st.session_state.image_results)
            else:
                st.info(
                    "No reference photos provided. Upload reference images in the sidebar to see matching outfits here.")
