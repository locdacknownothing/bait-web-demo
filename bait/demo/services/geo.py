# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from exif import Image

from time import sleep

from streamlit.components.v1 import html
from streamlit_cookies_controller import CookieController

from apis.map_api import get_user_token, get_reverted_geocode


def fetch_ip_info_to_cookies():
    js_code = f"""
        <script>
            function getCookie(name) {{
                const value = `; ${{document.cookie}}`;
                const parts = value.split(`; ${{name}}=`);
                if (parts.length === 2) return parts.pop().split(';').shift();
            }}

            function setCookie(name, value, days) {{
                const d = new Date();
                d.setTime(d.getTime() + (days * 24 * 60 * 60 * 1000));
                let expires = "expires="+d.toUTCString();
                document.cookie = name + "=" + value + ";" + expires + ";path=/";
            }}

            function success(position) {{
                const latitude = position.coords.latitude;
                const longitude = position.coords.longitude;
            
                let cookieName = 'user_location';
                let cookieValue = getCookie(cookieName);

                setCookie('user_location', JSON.stringify({{'lat': latitude, 'lng': longitude}}), 1);
            }}

            function error() {{
                alert("Sorry, no position available.");
            }}
             
            // navigator.geolocation.getCurrentPosition(success, error);

            const options = {{
                enableHighAccuracy: true,
                maximumAge: 30000,
                timeout: 27000,
            }};

            const watchID = navigator.geolocation.watchPosition(success, error, options);
        </script>
    """
    html(js_code, height=0)


def get_client_gps_coordinates():
    # g = geocoder.ip(location="me")
    # g_json = g.response.json()
    # location = g_json.get("loc", "")
    # if location:
    #     return list(map(float, location.split(",")))
    # else:
    #     return None

    cookie_controller = CookieController()
    location = cookie_controller.get("user_location")
    print(location)
    return location
    

def geo_reverse(lat, lng):
    try:
        token = get_user_token()
        reverted_geocode_response = get_reverted_geocode(token, lat, lng)
        if reverted_geocode_response is None:
            return None

        data = reverted_geocode_response.get("data", None)
        if data is None:
            return None

        a4_parts = [data.get("a4_prefix", ""), data.get("a4_name", "")]
        a3_parts = [data.get("a3_prefix", ""), data.get("a3_name", "")]
        a2_parts = [data.get("a2_prefix", ""), data.get("a2_name", "")]
        
        address_parts = list(map(
            lambda ax: " ".join([part for part in ax if part]), 
            [a4_parts, a3_parts, a2_parts]
        ))
        address_parts = [part for part in address_parts if part]
        return ", ".join(address_parts)
    except Exception as e:
        print(str(e))


def get_current_address():
    
    location = get_client_gps_coordinates()
    if isinstance(location, dict):
        if location.get("lat") and location.get("lng"):
            return geo_reverse(location["lat"], location["lng"])
    else:
        return None
    


def dms_coordinates_to_dd_coordinates(coordinates, coordinates_ref):
    decimal_degrees = coordinates[0] + \
                      coordinates[1] / 60 + \
                      coordinates[2] / 3600
    
    if coordinates_ref == "S" or coordinates_ref == "W":
        decimal_degrees = -decimal_degrees
    
    return decimal_degrees

def get_coor_from_metadata_images(img_path):
    '''
    Lấy coor từ metadata của hình
    '''

    with open(img_path, 'rb') as img_file:
        img = Image(img_file)
        if not img.has_exif:
            return None
        else:
            lat = dms_coordinates_to_dd_coordinates((img.gps_latitude),img.gps_latitude_ref)
            long = dms_coordinates_to_dd_coordinates((img.gps_longitude),img.gps_longitude_ref)
            # lat = img.gps_latitude
            # long = img.gps_longitude
            return geo_reverse(lat,long)
        