import gc
import os
import unittest

import zarr

from helper import utils


def _open_root():
    if not utils.PATH:
        raise unittest.SkipTest("utils.PATH is empty")
    try:
        return zarr.open_group(utils.PATH, mode="r")
    except Exception as exc:
        raise unittest.SkipTest(f"Unable to open Zarr at {utils.PATH}: {exc}") from exc


def _labels_for_product(root, product, max_labels=3):
    group_path = utils._PRODUCTS[product]["group"]
    group = root[group_path]
    labels_arr = group["labels"]
    count = min(int(labels_arr.shape[0]), max_labels)
    return [utils._normalize_label(v) for v in labels_arr[:count]]


class TestHelperUtils(unittest.TestCase):
    def _log_mem(self, label):
        rss_kb = None
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as fp:
                for line in fp:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            rss_kb = int(parts[1])
                        break
        except Exception:
            rss_kb = None
        if rss_kb is not None:
            print(f"[MEM] {label} rss_kb={rss_kb}")
        else:
            print(f"[MEM] {label} rss_kb=unknown pid={os.getpid()}")

    def tearDown(self):
        self._log_mem("before_gc")
        gc.collect()
        self._log_mem("after_gc")

    def _run_product_test(self, name, getter):
        print(f"\n[TEST] product={name} start")
        self._log_mem("start")
        root = None
        labels = None
        result = None
        try:
            root = _open_root()
            labels = _labels_for_product(root, name)
            cadence = utils._PRODUCTS[name]["cadence"]

            if cadence == "static":
                result = getter(include_labels=True)
            else:
                if not labels:
                    self.skipTest(f"No labels for {name}")
                start = utils._label_to_date(labels[0], cadence)
                result = getter(start_date=start, end_date=start, include_labels=True)
            self._log_mem("after_getter")

            self.assertEqual(result["name"], name)
            self.assertEqual(result["cadence"], cadence)
            self.assertIn("data", result)
            self.assertIn("valid", result)
            self.assertIn("labels", result)
            self.assertIn("missing_labels", result)

            if result["labels"]:
                self.assertIsNotNone(result["data"])
                self.assertIsNotNone(result["valid"])
                self.assertTrue(hasattr(result["data"], "shape"))
                self.assertTrue(hasattr(result["valid"], "shape"))
                self.assertEqual(result["data"].shape[0], int(root[utils._PRODUCTS[name]["group"]]["data"].shape[0]))
                self.assertEqual(result["valid"].shape[0], int(root[utils._PRODUCTS[name]["group"]]["valid"].shape[0]))
                self.assertTrue(set(result["missing_labels"]).issubset(set(result["labels"])))
                if isinstance(result["selection"], slice):
                    self.assertGreaterEqual(result["selection"].stop or 0, result["selection"].start or 0)
                else:
                    self.assertEqual(len(result["selection"]), len(result["labels"]))
                label, data_slice, valid_slice = next(iter(utils.iter_product_slices(
                    name,
                    start_date=result["start_date"],
                    end_date=result["end_date"],
                )))
                self._log_mem("after_single_slice")
                self.assertIn(label, result["labels"])
                self.assertEqual(data_slice.shape, result["data"].shape[1:])
                self.assertEqual(valid_slice.shape, result["valid"].shape[1:])
                chunk_labels, chunk_data, chunk_valid = next(iter(utils.iter_product_chunks(
                    name,
                    start_date=result["start_date"],
                    end_date=result["end_date"],
                    chunk_len=1,
                )))
                self._log_mem("after_chunk")
                self.assertEqual(len(chunk_labels), 1)
                self.assertEqual(chunk_data.shape[0], 1)
                self.assertEqual(chunk_valid.shape[0], 1)
            if cadence == "static":
                self.assertIsNone(result["start_date"])
                self.assertIsNone(result["end_date"])
        finally:
            del root, labels, result
            self._log_mem("after_cleanup")
            print(f"[TEST] product={name} done")

    def test_sentinel2(self):
        self._run_product_test("sentinel2", utils.get_sentinel2)

    def test_sentinel1(self):
        self._run_product_test("sentinel1", utils.get_sentinel1)

    def test_landsat(self):
        self._run_product_test("landsat", utils.get_landsat)

    def test_era5(self):
        self._run_product_test("era5", utils.get_era5)

    def test_modis(self):
        self._run_product_test("modis", utils.get_modis)

    def test_viirs(self):
        self._run_product_test("viirs", utils.get_viirs)

    def test_alphaearth(self):
        self._run_product_test("alphaearth", utils.get_alphaearth)

    def test_dem(self):
        self._run_product_test("dem", utils.get_dem)

    def test_dynamic_world(self):
        self._run_product_test("dynamic_world", utils.get_dynamic_world)

    def test_worldcover(self):
        self._run_product_test("worldcover", utils.get_worldcover)

    def test_load_all_data(self):
        print("\n[TEST] load_all_data start")
        self._log_mem("start")
        root = _open_root()
        # Use the first available non-static product to avoid heavy loads.
        name = next((p for p, meta in utils._PRODUCTS.items() if meta["cadence"] != "static"), None)
        if name is None:
            self.skipTest("No time-varying products available.")
        labels = _labels_for_product(root, name)
        if not labels:
            self.skipTest(f"No labels for {name}")
        cadence = utils._PRODUCTS[name]["cadence"]
        start = utils._label_to_date(labels[0], cadence)
        results = utils.load_all_data(start_date=start, end_date=start, include_labels=False)
        self._log_mem("after_load_all_data")
        self.assertTrue(isinstance(results, dict))
        for prod in utils._PRODUCTS:
            self.assertIn(prod, results)
            self.assertEqual(results[prod]["name"], prod)
        del root, labels, results
        self._log_mem("after_cleanup")
        print("[TEST] load_all_data done")


if __name__ == "__main__":
    unittest.main()
