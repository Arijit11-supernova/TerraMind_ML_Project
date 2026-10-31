"""Diagnose access to the IMD rainfall archive."""

import requests

BASE_URL = "https://www.imdpune.gov.in/cmpg/Griddata"

session = requests.Session()

# Check whether the archive page is accessible.
page_url = f"{BASE_URL}/Rainfall_25_NetCDF.html"

try:
    response = session.get(page_url, timeout=30)

    print("ARCHIVE PAGE")
    print("Status:", response.status_code)
    print("Final URL:", response.url)
    print("Content-Type:", response.headers.get("Content-Type"))
except requests.RequestException as error:
    print("Archive page request failed:", error)

# Compare one year that previously worked with one that failed.
for year in (2003,):
    print(f"\nTEST YEAR: {year}")

    try:
        response = session.post(
            f"{BASE_URL}/RF25.php",
            data={"RF25": str(year)},
            timeout=(15, 60),
        )

        print("Status:", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type"))
        print("Response size:", len(response.content), "bytes")
        print("Final URL:", response.url)
        print("Response preview:", repr(response.text[:250])
              if "text" in response.headers.get("Content-Type", "").lower()
              else repr(response.content[:40]))

    except requests.RequestException as error:
        print("Request failed:", error)