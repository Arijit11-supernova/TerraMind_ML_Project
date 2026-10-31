import requests
import xarray as xr
import tempfile
import os

url = "https://www.imdpune.gov.in/cmpg/Griddata/RF25.php"

print("Downloading 2003 rainfall data...")

response = requests.post(
    url,
    data={"RF25": "2003"},
    timeout=(15, 60)
)

response.raise_for_status()

print("HTTP status:", response.status_code)
print("Response size:", len(response.content), "bytes")
print("File signature:", response.content[:4])

temp_path = None

try:
    with tempfile.NamedTemporaryFile(
        suffix=".nc",
        delete=False
    ) as temp_file:
        temp_file.write(response.content)
        temp_path = temp_file.name

    print("\nOpening NetCDF file...")

    with xr.open_dataset(temp_path) as ds:
        print(ds)
        print("\nFirst date:", str(ds["TIME"].values[0]))
        print("Last date:", str(ds["TIME"].values[-1]))
        print("Number of dates:", ds.sizes["TIME"])
        print("Rainfall units:", ds["RAINFALL"].attrs.get("units"))

finally:
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)