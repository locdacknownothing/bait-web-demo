import requests
from urllib.parse import urlencode, urljoin


def call_api(url, headers=None, data=None, method="GET"):
    """
    Make an API call to the given URL with the given headers and data.

    Args:
        url (str): The URL of the API endpoint to call.
        headers (dict): The headers to send with the request.
        data (dict): The Python dictionary data to send with the request.
        method (str): The HTTP method to use for the request. Defaults to "GET".

    Returns:
        The JSON response from the API, or None if the request failed.
    """
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, data=data)
        elif method == "PUT":
            response = requests.put(url, headers=headers, data=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, data=data)
        else:
            raise ValueError(f"Invalid HTTP method: {method}")

        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None


def build_url(api_base_url, endpoint, params=None):
    """
    Build a URL by joining the base URL and endpoint and appending any query parameters.

    Args:
        api_base_url (str): The base URL of the API.
        endpoint (str): The endpoint of the API.
        params (dict): A dictionary of query parameters to append to the URL.

    Returns:
        str: The full URL.
    """
    url = urljoin(api_base_url, endpoint)

    if params:
        encoded_params = urlencode(params)
        if encoded_params:
            url += f"?{encoded_params}"

    return url
