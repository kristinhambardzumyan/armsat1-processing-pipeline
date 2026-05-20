from pathlib import Path
import ee
import requests

def init_gee(project: str | None = None):
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()

def build_s2_rgb_mosaic(bbox_wgs84, time_interval, max_cloud_pct=100, composite="least_cloudy_mosaic"):
    region = ee.Geometry.Rectangle(list(bbox_wgs84))

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(time_interval[0], time_interval[1])
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .sort("CLOUDY_PIXEL_PERCENTAGE", False)
    )

    img = collection.mosaic()

    rgb = (
        img.select(["B4", "B3", "B2"])
        .multiply(6.5535)
        .clamp(1, 65535)
        .uint16()
        .clip(region)
        .unmask(0)
    )

    return rgb, region

def download_mosaic_rgb_to_local(
    scene: str,
    bbox_wgs84,
    time_interval,
    out_path: Path,
    resolution=10,
    max_cloud_pct=100,
    composite="least_cloudy_mosaic",
):
    image, region = build_s2_rgb_mosaic(bbox_wgs84, time_interval, max_cloud_pct, composite)

    url = image.getDownloadURL({
        "region": region,
        "scale": resolution,
        "format": "GEO_TIFF",
    })

    print(f"\nDownloading {scene} locally...")

    r = requests.get(url, stream=True)
    r.raise_for_status()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    return str(out_path)

def export_mosaic_rgb_to_drive(
    scene: str,
    bbox_wgs84,
    time_interval,
    resolution=10,
    max_cloud_pct=100,
    composite="least_cloudy_mosaic",
    drive_folder="armsat_sentinel_exports",
):
    image, region = build_s2_rgb_mosaic(bbox_wgs84, time_interval, max_cloud_pct, composite)

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"{scene}_s2_rgb",
        folder=drive_folder,
        fileNamePrefix=f"{scene}_sentinel2_rgb_raw",
        region=region,
        scale=resolution,
        fileFormat="GeoTIFF",
        maxPixels=1e13,
    )

    task.start()
    print(f"\nStarted Drive export for {scene}")
    print(f"Task ID: {task.id}")
    print(f"Drive folder: {drive_folder}")
    return task.id