import streamlit as st
from apis.weights_api import download_and_extract_zip

@st.cache_resource
def download_weights():
    download_and_extract_zip(st.secrets.weights.base_url, "./")
