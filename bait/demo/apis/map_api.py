# from os import getenv
# from dotenv import load_dotenv

from apis.core import build_url, call_api


# load_dotenv()  # Load variables from the .env file

# api_base_url = getenv("API_MAP_BASE_URL")
# api_username = getenv("API_MAP_USERNAME")
# api_password = getenv("API_MAP_PASSWORD")

import streamlit as st


api_base_url = st.secrets["API_MAP_BASE_URL"]
api_username = st.secrets["API_MAP_USERNAME"]
api_password = st.secrets["API_MAP_PASSWORD"]


def get_user_token():
    endpoint = "/user/token"
    url = build_url(api_base_url, endpoint)

    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "grant_type": "password",
        "username": api_username,
        "password": api_password,
    }

    res = call_api(url, headers=headers, data=data, method="POST")
    return res["access_token"]


def get_reverted_geocode(token, lat, lng):
    endpoint = "/admin/revert"
    params = {"lat": lat, "lng": lng}
    url = build_url(api_base_url, endpoint, params=params)

    headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}

    res = call_api(url, headers=headers, method="GET")

    return res
