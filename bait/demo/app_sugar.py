from pathlib import Path
import streamlit as st
from helpers import create_temp_dir, image_path_to_base64
from services_sugar import (
    get_annotated_image,
    get_detections_data,
    od_traffic,
    save_uploaded_image,
)
from ui import centered_button

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlock"]
    {
        display: flex;
        flex-direction: column; /* Ensure items are stacked vertically */
        justify-content: center; /* Vertically center items */
        align-items: center; /* Horizontally center items (optional) */
        text-align:center,
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    if "image_path" not in st.session_state:
        st.session_state.image_path = None

    if "analysis_triggered" not in st.session_state:
        st.session_state.analysis_triggered = False

    if "detections" not in st.session_state:
        st.session_state.detections = None

    if "selected_row_id" not in st.session_state:
        st.session_state.selected_row_id = None


def clear_session_state():
    st.session_state.uploaded_file = None
    st.session_state.image_path = None
    st.session_state.analysis_triggered = False
    st.session_state.selected_row_id = None
    st.session_state.detections = None


def trigger_analysis():
    st.session_state.analysis_triggered = True


def trigger_selection(selected_id):
    st.session_state.selected_row_id = selected_id


init_session_state()

st.logo(image="assets/logo.png", size="large")
st.title("DGM AI Demo")

if st.session_state.uploaded_file is None:
    uploaded_file = st.file_uploader("Upload an image")
    print(uploaded_file)

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.rerun()
else:
    st.button("Upload another image", on_click=clear_session_state)
    output_dir = create_temp_dir(".streamlit/static/tmp")
    image_path = save_uploaded_image(st.session_state.uploaded_file, output_dir)

    if not st.session_state.analysis_triggered:
        if image_path is not None:
            st.image(image_path, caption="Uploaded Image")
            centered_button("Analyze", type="primary", on_click=trigger_analysis)
    else:
        # st.write(image_path)
        if not st.session_state.detections:
            detections = od_traffic(image_path, output_dir=output_dir)
            st.session_state.detections = detections

        annotated_image = get_annotated_image(image_path, output_dir)
        st.image(annotated_image, caption="Analyzed Image")

        df = get_detections_data(st.session_state.detections, output_dir)
        cropped_image_paths = df["View"]
        df["View"] = df["View"].apply(lambda x: image_path_to_base64(x))

        # print(df)
        st.write("### Object Detection")
        table_data = st.dataframe(
            df,
            column_config={"View": st.column_config.ImageColumn()},
            hide_index=True,
            key="data_editor",
            use_container_width=True,
        )

        selected_row_id = st.session_state.get("selected_row_id", None)
        selected_id = st.number_input(
            label="Select Row ID to view details",
            min_value=0,
            max_value=len(df) - 1 if not df.empty else 0,
            value=selected_row_id if selected_row_id is not None else 0,
        )

        if not df.empty and 0 <= selected_id < len(df):
            # st.write(selected_id)
            cropped_image_path = cropped_image_paths[selected_id]
            selected_detection = df.iloc[selected_id]
            label = selected_detection["Label"]

            st.subheader(f"Label Details")
            st.image(cropped_image_path, caption=f"{label}")
            if label == "POI":
                pass
            else:
                pass
