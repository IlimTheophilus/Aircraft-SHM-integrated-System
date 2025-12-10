# Python
import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO

# Function to fetch and open image from URL


def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error if the request failed

        # Convert content to BytesIO object for PIL
        image_data = BytesIO(response.content)

        # Open image using PIL
        image = Image.open(image_data)
        return image

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the image: {e}")
        return None
    except UnidentifiedImageError:
        st.error("The image cannot be identified. Check the URL or image format.")
        return None


# Example URL (replace with your own)
image_url = "https://th.bing.com/th/id/OIP.sf0jcoZzEDF5xNM353T4xwHaES?w=314&h=182&c=7&r=0&o=7&pid=1.7&rm=3"

# Load and display the image
image = load_image_from_url(image_url)
if image is not None:
    st.image(image, caption="Loaded Image from URL")
