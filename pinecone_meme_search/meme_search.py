import os
import clip
import torch
from PIL import Image
import requests
from io import BytesIO
from pinecone import Pinecone, PodSpec

from dotenv import load_dotenv
_ = load_dotenv()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def setup_pinecone_index():
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = "starter-index-meme-search"

    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)

    # Create a Pinecone index with the appropriate environment for the starter tier
    # Since we are using CLIP which produces 512-dimensional vectors, we'll set the dimension to 512.
    if index_name not in  pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=512,
            metric="cosine",
            spec=PodSpec(environment="gcp-starter")
        )

    # Connect to the newly created or existing index
    meme_index = pc.Index(name=index_name)

    return meme_index


def process_and_upload_image(image_url, index):
    try:
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        image = Image.open(BytesIO(response.content))
        
        # Preprocess the image and generate embedding
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_embedding = image_features.cpu().numpy().tolist()[0]
        
        # The upsert method updates the vector if the id already exists, preventing duplicates.
        index.upsert(vectors=[(image_url, image_embedding)])
        
    except requests.exceptions.RequestException as e:
        # This will catch HTTPError and other request exceptions
        print(f"Failed to download {image_url}: {e}")
    except Exception as e:
        # This will catch other exceptions, such as errors during image preprocessing or upserting.
        print(f"An error occurred for {image_url}: {e}")

def search_meme(query, index, top_k = 1):
    # Convert query to embedding
    text_input = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_input)
        text_embedding = text_features.cpu().numpy().tolist()[0]
    
    # Search in Pinecone
    try:
        results = index.query(vector=[text_embedding], top_k=top_k)
    except Exception as e:
        print(f"An error occurred: {e}")
    
    for match in results["matches"]:
        print(f"URL: {match['id']}, Score: {match['score']}")
    
    return results["matches"]