import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from apis.core import call_api


def get_public_ip():
    res = call_api("https://api64.ipify.org/?format=json")
    return res


def get_ip_info():
    res = call_api("https://ipinfo.io")
    return res


if __name__ == "__main__":
    print(get_public_ip())
    print(get_ip_info())
    