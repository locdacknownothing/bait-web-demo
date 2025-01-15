from pathlib import Path


def save_uploaded_image(uploaded_file, output_dir):
    uploaded_image_path = Path(output_dir) / uploaded_file.name
    with open(uploaded_image_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    return uploaded_image_path
