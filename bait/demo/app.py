import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import streamlit as st
from helpers import (
    create_temp_dir, 
    delete_temp_dir, 
    image_path_to_base64,
    read_image_from_bytes,
    check_image_resolution
)
from services.od_poi_pa import (
    get_annotated_image,
    get_detections_data,
    od,
)

from services.od_traffic_sign import (
    get_annotated_image_traffic,
    get_detections_data_traffic,
    od_traffic,
)

from services.upload import save_uploaded_image
from services.text import get_poi_data, get_traffic_sign_data
from services.geo import fetch_ip_info_to_cookies, get_current_address, get_coor_from_metadata_images
from services.weights import download_weights
from ui import centered_button, traffic_button


def init_session_state():
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    if "image_path" not in st.session_state:
        st.session_state.image_path = None

    if "analysis_triggered" not in st.session_state:
        st.session_state.analysis_triggered = False

    if "analysis_triggered_traffic_sign" not in st.session_state:
        st.session_state.analysis_triggered_traffic_sign = False

    if "detections" not in st.session_state:
        st.session_state.detections = None

    if "selected_row_id" not in st.session_state:
        st.session_state.selected_row_id = None


def clear_session_state():
    st.session_state.uploaded_file = None
    st.session_state.image_path = None
    st.session_state.analysis_triggered = False
    st.session_state.analysis_triggered_traffic_sign = False
    st.session_state.selected_row_id = None
    st.session_state.detections = None


def trigger_analysis():
    st.session_state.analysis_triggered = True

def trigger_analysis_traffic_sign():
    st.session_state.analysis_triggered_traffic_sign = True


def trigger_selection(selected_id):
    st.session_state.selected_row_id = selected_id


init_session_state()

st.logo(image="assets/logo.png", size="large")
st.title("DGM AI Demo")

resolution_threshold = (1440, 1080)
output_dir = ".streamlit/static/tmp"


if st.session_state.uploaded_file is None:
    uploaded_file = st.file_uploader("Upload an image")

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = read_image_from_bytes(image_bytes)
        
        if check_image_resolution(image, resolution_threshold):
            st.session_state.uploaded_file = uploaded_file
            st.rerun()
            # delete_temp_dir(output_dsir)
        else:
            st.error(f"Please upload an image with a resolution of at least {resolution_threshold[0]}x{resolution_threshold[1]}.")
else:
    output_dir = create_temp_dir(output_dir)
    st.button(
        "Upload another image", 
        on_click=lambda: (
            clear_session_state(),
            delete_temp_dir(output_dir)
        )
    )
    image_path = save_uploaded_image(st.session_state.uploaded_file, output_dir)

    if not st.session_state.analysis_triggered and not st.session_state.analysis_triggered_traffic_sign:
        if image_path is not None:
            st.image(image_path, caption="Uploaded Image")
            download_weights()
            centered_button("Analyze Building Information", type="primary", on_click=trigger_analysis)
            traffic_button("Analyze Traffic Sign", type="primary", on_click=trigger_analysis_traffic_sign)
    
    elif st.session_state.analysis_triggered:
        
        if not st.session_state.detections:
            detections = od(image_path, output_dir=output_dir)
            st.session_state.detections = detections

        annotated_image = get_annotated_image(image_path, output_dir)
        st.image(annotated_image, caption="Analyzed Image")

        st.write("### Object Detection")
        df = get_detections_data(st.session_state.detections, output_dir)

        if df is None:
            st.write("No detections found")
        else:

            cropped_image_paths = df["View"]
            df["View"] = df["View"].apply(lambda x: image_path_to_base64(x))

            table_data = st.dataframe(
                df,
                column_config={"View": st.column_config.ImageColumn()},
                hide_index=True,
                key="data_editor",
                use_container_width=True,
            )

            # selected_row_id = st.session_state.get("selected_row_id", None)
            # selected_id = st.number_input(
            #     label="Select row ID to view details",
            #     min_value=0,
            #     max_value=len(df) - 1 if not df.empty else 0,
            #     value=selected_row_id if selected_row_id is not None else 0,
            # )

            selected_id = st.selectbox(
                "Select row ID to view details",
                tuple(range(len(df))),
            )

            if not df.empty and 0 <= selected_id < len(df):
                # st.write(selected_id)
                cropped_image_path = cropped_image_paths[selected_id]
                selected_detection = df.iloc[selected_id]
                label = selected_detection["Label"]

                st.subheader(f"Label Details")
                st.image(cropped_image_path, caption=f"{label}")
                if label == "POI":
                    with st.spinner(f"Analyzing {label} Information..."):
                        poi_data = get_poi_data(cropped_image_path)

                        if poi_data.get("name"):
                            st.write(f"Name: {poi_data['name']}")

                        if poi_data.get("cate") is not None:
                            #st.write(f"Category: {poi_data['cate']}")
                            st.write(f"Cate")
                            st.dataframe(poi_data.get("cate"))

                        if poi_data.get("address_number"):
                            st.write(f"Address Number: {poi_data['address_number']}")

                    # with st.spinner(f"Getting POI location ..."):
                        #address = get_current_address()

                        ## Lấy coor từ tấm ảnh
                        address = get_coor_from_metadata_images(image_path)
                        if address:
                            st.write(f"Address: {address}")

                        # if poi_data.get("address"):
                        #     st.write(f"Full address: {poi_data['address']}")

                        # st.write(f"OCR: {poi_data['ocr']}")
                        # st.json(poi_data["ner"])
                else:
                    # TODO: add data for PA, if having time
                    pass
    else:
        if not st.session_state.detections:
            detections = od_traffic(image_path, output_dir=output_dir)
            st.session_state.detections = detections
        
        annotated_image = get_annotated_image_traffic(image_path, output_dir)
        st.image(annotated_image, caption="Analyzed Image")

        st.write("### Object Detection")
        df = get_detections_data_traffic(st.session_state.detections, output_dir)

        if df is None:
            st.write("No detections found")
        else:
            cropped_image_paths = df["View"]
            df["View"] = df["View"].apply(lambda x: image_path_to_base64(x))

            table_data = st.dataframe(
                df,
                column_config={"View": st.column_config.ImageColumn()},
                hide_index=True,
                key="data_editor",
                use_container_width=True,
            )


            selected_id = st.selectbox(
                "Select row ID to view details",
                tuple(range(len(df))),
            )

            if not df.empty and 0 <= selected_id < len(df):
                # st.write(selected_id)
                cropped_image_path = cropped_image_paths[selected_id]
                selected_detection = df.iloc[selected_id]
                label = selected_detection["Label"]

                st.subheader(f"Label Details")
                st.image(cropped_image_path, caption=f"{label}")


                # sign_data = get_traffic_sign_data(cropped_image_path)
                # st.write(sign_data)
