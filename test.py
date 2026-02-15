# inspect_target_crs.py
import zarr
from pathlib import Path

ZARR_PATH = Path("/home/naren-root/Documents/FYP2/Project/madurai.zarr")


def iter_groups(g, prefix=""):
    yield prefix, g
    for name, item in g.groups():
        sub = f"{prefix}/{name}" if prefix else name
        yield from iter_groups(item, sub)


def main():
    root = zarr.open_group(str(ZARR_PATH), mode="r")

    rows = []
    for path, grp in iter_groups(root):
        attrs = dict(grp.attrs)
        if "crs" in attrs:
            rows.append(
                {
                    "path": path or "/",
                    "crs": attrs.get("crs"),
                    "transform": attrs.get("transform"),
                    "width": attrs.get("width"),
                    "height": attrs.get("height"),
                    "pixel_size_x": attrs.get("pixel_size_x"),
                    "pixel_size_y": attrs.get("pixel_size_y"),
                    "reference_file": attrs.get("reference_file"),
                }
            )

    if not rows:
        print("No groups with a CRS attribute found.")
        return

    for r in rows:
        print(f"[{r['path']}]")
        print(f"  crs: {r['crs']}")
        print(f"  transform: {r['transform']}")
        print(f"  width,height: {r['width']},{r['height']}")
        print(f"  pixel_size_x,y: {r['pixel_size_x']},{r['pixel_size_y']}")
        print(f"  reference_file: {r['reference_file']}")
        print()


if __name__ == "__main__":
    main()
