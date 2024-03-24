import streamlit as st
import requests
from PIL import Image
from io import BytesIO

from pinecone_meme_search.meme_search import search_meme, setup_pinecone_index

# Initialize your Pinecone index
meme_index = setup_pinecone_index()

st.set_page_config(page_title="Meme Search", page_icon="🐶")

st.sidebar.header("Meme Search-Engine")
st.sidebar.write("This is multimodal search powered by Pinecone and CLIP, to find the most semantically related meme based on your query.")

st.title('Meme Search App')
# st.write('''
#     The search is powered by Pinecone and CLIP, finding the most semantically related memes based on your query.
# ''')

query = st.text_input('Enter a search query for a meme:')

if query:
    # Placeholder for search function - implement this based on your project
    # For now, we're simulating a response
    search_results = search_meme(query, index=meme_index)

    if search_results:
        # Assuming the first result is the most relevant
        meme_url = search_results[0]['id']
        response = requests.get(meme_url)
        image = Image.open(BytesIO(response.content))
        # st.image(image, caption='Da Meme', width=300)
        st.columns(3)[1].image(image, caption='Da Meme', width=300)
    else:
        st.write('No memes found for this query.')