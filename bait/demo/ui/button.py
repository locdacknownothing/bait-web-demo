import streamlit as st


def centered_button(*args, **kwargs):
    middle = st.columns(3)[1]
    middle.button(*args, **kwargs, use_container_width=True)


def traffic_button(*args, **kwargs):
    middle = st.columns(3)[1]
    return middle.button(*args, **kwargs, use_container_width=True)
