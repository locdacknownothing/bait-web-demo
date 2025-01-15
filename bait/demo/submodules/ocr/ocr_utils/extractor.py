import re
import numpy as np
import cv2
from unidecode import unidecode

from ocr_utils.process import process_list
from recognition_result import RegRes


vocab = "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789 "


def filter_characters(text: str, vocab: str) -> str:
    """Remove characters not in vocabulary"""
    pattern = f"^[^{re.escape(vocab)}]+|[^{re.escape(vocab)}]+$"
    text = re.sub(pattern, "", text)
    return text


def filter_text(result_value: list, vocab: str) -> list:
    """Filter text with characters in vocabulary

    Args:
        result_value (list): a list of recognition result objects.
        vocab (str): a string contains valid characters

    Returns:
        list: a filtered list of `result_value`
    """
    for dr in result_value:
        dr.text = filter_characters(dr.text, vocab)

    result_value = [x for x in result_value if x.text]
    return result_value


# Is deprecated, can be used for skewed images
def is_same_row(
    bbox1: list | tuple, bbox2: list | tuple, bbox3: list | tuple | None
) -> bool:
    """Check if current bbox1 is on the same row as previous bbox2"""

    h1 = bbox1[3] - bbox1[1]
    h2 = bbox2[3] - bbox2[1]
    cx1 = (bbox1[2] + bbox1[0]) / 2
    cy1 = (bbox1[3] + bbox1[1]) / 2
    cx2 = (bbox2[2] + bbox2[0]) / 2
    cy2 = (bbox2[3] + bbox2[1]) / 2

    vdist = abs(cy1 - cy2)
    if not 0.75 * h2 <= h1 <= 1.25 * h2:
        return False

    if vdist <= 0.5 * h1 or vdist <= 0.5 * h2:
        return True

    if vdist <= 0.75 * h1 and vdist <= 0.75 * h2:
        if bbox3 is None:
            return True
        else:
            cx3 = (bbox3[2] + bbox3[0]) / 2
            prev_hdist = cx2 - cx3
            curr_hdist = cx1 - cx2
            return 0.9 * prev_hdist <= curr_hdist <= 1.1 * prev_hdist

    return False


# Is deprecated
def sort_by_bounding_boxes(data: list):
    """Sort a list of data by bounding box positions

    Args:
        data(list[RegRes]): a list of recognition result objects.

        Example:
        ```python
            [
                <RegRes(
                    xyxy: <[x_min, y_min, x_max, y_max]>,
                    text: <text>,
                    ...
                )>, ...
            ]
        ```
    Returns:
        list[list[RegRes]]: a list of sub-lists, each sub-list is considered as a
            row of recognition result objects.
    """
    if not data:
        return []

    # Sort by the y_min value
    data = sorted(data, key=lambda item: item.xyxy[1])

    # Group by rows
    rows = []
    current_row = []
    current_bbox = data[0].xyxy
    prev_bbox = None

    for i, item in enumerate(data):
        bbox = item.xyxy

        if is_same_row(bbox, current_bbox, prev_bbox):
            current_row.append(item)
            if i >= 1:
                prev_bbox = data[i - 1].xyxy
        else:
            rows.append(current_row)
            current_row = [item]
            prev_bbox = None

        current_bbox = bbox

    # Don't forget to add the last row
    rows.append(current_row)

    # Sort all rows by the x_min value (left)
    for i, row in enumerate(rows):
        rows[i] = sorted(row, key=lambda item: item.xyxy[0])

    # sorted_data = [bbox for row in rows for bbox in row]
    return rows


def is_on_row(bbox1, row):
    """Check if bounding box is on the row"""
    if not row:
        raise ValueError("Cannot check an empty row of boxes.")

    # Get vertically-nearest box
    bbox2 = row[-1].xyxy
    h1 = bbox1[3] - bbox1[1]
    h2 = bbox2[3] - bbox2[1]
    cy1 = (bbox1[3] + bbox1[1]) / 2
    cy2 = (bbox2[3] + bbox2[1]) / 2

    vsim1 = abs(h1 - h2) < 0.5 * max(h1, h2)
    vsim2 = abs(cy1 - cy2) < 0.25 * max(h1, h2)

    return vsim1 and vsim2


def sort_by_bounding_boxes_v2(data: list):
    """Sort a list of data by bounding box positions

    Args:
        data(list[RegRes]): a list of recognition result objects.

        Example:
        ```python
            [
                <RegRes(
                    xyxy: <[x_min, y_min, x_max, y_max]>,
                    text: <text>,
                    ...
                )>, ...
            ]
        ```
    Returns:
        list[list[RegRes]]: a list of sub-lists, each sub-list is considered as a
            row of recognition result objects.
    """
    if not data:
        return []

    # Sort by the y_min value
    data = sorted(data, key=lambda item: item.xyxy[1])

    # Group by rows
    rows = []

    for index, item in enumerate(data):
        if not rows:
            rows.append([item])
            continue

        bbox = item.xyxy
        is_appended = False

        for row in rows[::-1]:
            if is_on_row(bbox, row):
                row.append(item)
                is_appended = True
                break

        if not is_appended:
            rows.append([item])

    # Sort all rows by the x_min value (left)
    for i, row in enumerate(rows):
        rows[i] = sorted(row, key=lambda item: item.xyxy[0])

    return rows


def get_height(box):
    return box[3] - box[1]


def get_center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def _is_height_similar_(box, row, height_threshold_factor=0.5):
    avg_line_height = sum(get_height(b) for b in row) / len(row)
    height_threshold = avg_line_height * height_threshold_factor

    is_height_similar = all(
        abs(get_height(box) - get_height(current_box)) <= height_threshold
        for current_box in row
    )

    return is_height_similar


def _is_vertically_aligned_(box, row, line_alignment_threshold_factor=0.5):
    avg_line_height = sum(get_height(b) for b in row) / len(row)
    line_alignment_threshold = avg_line_height * line_alignment_threshold_factor

    if row:
        line_points = np.array([get_center(b) for b in row], dtype=np.float32)
        if len(line_points) >= 2:  # Need at least two points to fit a line
            line_params = cv2.fitLine(line_points, cv2.DIST_L2, 0, 0.001, 0.001)
            vx, vy, x0, y0 = line_params.flatten()
            x, y = get_center(box)
            distance = np.abs((x - x0) * vy - (y - y0) * vx)
            is_vertically_aligned = distance <= line_alignment_threshold
        elif (
            len(line_points) == 1
        ):  # If only one point, consider alignment based on vertical proximity
            current_center_y = line_points[0][1]
            box_center_y = get_center(box)[1]
            is_vertically_aligned = (
                abs(box_center_y - current_center_y) <= line_alignment_threshold
            )
        else:
            is_vertically_aligned = True  # For the first box in a line

    return is_vertically_aligned


def sort_by_bounding_boxes_v3(data: list[RegRes]) -> list[list[RegRes]]:
    """
    Sort a list of RegRes objects by their bounding box positions.

    Args:
        data: A list of RegRes objects to sort.

    Returns:
        A list of lists of RegRes objects. Each sublist is a line of text sorted
        by the x-coordinate of the bounding box.
    """
    if not data:
        return []

    lines = []
    unassigned_boxes = list(data)

    while unassigned_boxes:
        current_line = [unassigned_boxes.pop(0)]
        remaining_boxes = list(unassigned_boxes)

        while True:
            added_to_line = False
            for box in list(remaining_boxes):
                box_ = box.xyxy
                current_line_ = [b.xyxy for b in current_line]

                is_height_similar = _is_height_similar_(box_, current_line_)
                is_vertically_aligned = _is_vertically_aligned_(box_, current_line_)

                if is_vertically_aligned and is_height_similar:
                    current_line.append(box)
                    unassigned_boxes.remove(box)
                    remaining_boxes.remove(box)
                    added_to_line = True

            if not added_to_line:
                break

        lines.append(sorted(current_line, key=lambda b: b.xyxy[0]))

    lines.sort(key=lambda line: min(box.xyxy[1] for box in line))
    return lines


def filter_names(rows: list[list], height_thresh: float = 0.8) -> list:
    """Filter rows of objects considered as POI's names into a list

    Args:
        rows (list[list]): a list of sub-lists, each sub-list is considered as a
            row of recognition result objects.

    Returns:
        list: a list of recognition result objects with only name text.
    """

    def max_height_row(row):
        if row:
            return max(row, key=lambda x: x.height).height
        else:
            return 0

    max_height_rows = [max_height_row(row) for row in rows]
    max_height = max(max_height_rows, default=0)
    result = []
    for i, row in enumerate(rows):
        if max_height_rows[i] >= height_thresh * max_height:
            result.extend(row)

    return result


def filter_by_height(reg_res_list: list, height_thresh: float = 0.8) -> list[str]:
    heights = [x.height for x in reg_res_list]
    max_height = max(heights, default=0)

    return [x for x in reg_res_list if x.height >= height_thresh * max_height]


def extract_names(result_value: list) -> list[str]:
    """Combined method to extract names from list of objects."""
    rv_rows = sort_by_bounding_boxes(result_value)
    rv_rows = [filter_text(row, vocab) for row in rv_rows]

    names = filter_names(rv_rows)
    if names:
        return [x.text for x in names]
    else:
        return []


def extract_important_text(result_value: list, height_thresh: float = 0.8) -> list[str]:
    """Extract important text from list of objects."""
    rv = filter_by_height(result_value, height_thresh)
    rv = filter_text(rv, vocab)
    rv_rows = sort_by_bounding_boxes(rv)

    return [x.text for row in rv_rows for x in row]


def extract_text_dict(result_dict: dict) -> dict:
    return {key: [x.text for x in value] for key, value in result_dict.items()}


def extract_string_dict(result_dict: dict) -> dict:
    text_list_dict = extract_text_dict(result_dict)
    str_dict = {
        key: " ".join(process_list(value)) for key, value in text_list_dict.items()
    }
    return str_dict


def extract_sorted_string_dict(result_dict: dict) -> dict:
    string_output = {}

    for image, results in result_dict.items():
        results = filter_text(results, vocab)
        sorted_results = [
            res for row_res in sort_by_bounding_boxes_v2(results) for res in row_res
        ]
        sorted_text = [res.text for res in sorted_results]
        sorted_string = " ".join(sorted_text)
        string_output[str(image)] = sorted_string

    return string_output


def extract_string(
    results_dict: dict[str, list[RegRes]],
    is_sorted: bool = False,
    is_lowercase: bool = False,
    apply_unidecode: bool = False,
) -> dict[str, str]:
    """
    Extract text from a dictionary of RegRes objects.

    Parameters
    ----------
    results_dict : dict[str, list[RegRes]]
        A dictionary where the keys are image paths and the values are lists of RegRes objects.
    is_sorted : bool, optional
        If True, sort the RegRes objects by their bounding boxes before converting them to a string.
        By default, False.
    is_lowercase : bool, optional
        If True, convert the extracted text to lowercase. By default, False.
    apply_unidecode : bool, optional
        If True, apply the unidecode function to the extracted text to remove accents. By default, False.

    Returns
    -------
    dict[str, str]
        A dictionary where the keys are the same as the input dictionary and the values are the
        extracted text strings.
    """

    output = {}

    for key, results in results_dict.items():
        results = filter_text(results, vocab)

        if is_sorted:
            results = [
                res for row_res in sort_by_bounding_boxes_v3(results) for res in row_res
            ]

        # Convert list of RegRes objects to list of strings
        results = " ".join([res.text for res in results])

        if is_lowercase:
            results = results.lower()

        if apply_unidecode:
            results = unidecode(results)

        output[str(key)] = results

    return output
