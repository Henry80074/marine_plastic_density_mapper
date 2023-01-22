import numpy as np
import shapely
from global_land_mask import globe
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import os
from dotenv import load_dotenv
from shapely.geometry import MultiPoint

from sentinel_downloader.weather import check_wind
from utils.dir_management import base_path, unzip_files
from datetime import datetime
from collections import OrderedDict

from utils.geographic_utils import get_min_max_long_lat

load_dotenv()

user_name = os.environ.get('USER_NAME')
password = os.environ.get('PASSWORD')
weather_key = os.environ.get('WEATHER')


class SentinelLoader(object):

    def __init__(self, start_date, end_date, max_cloud_percentage, max_wind_speed, tile_id=None):
        # connect to api
        self.cloud_percentage = (0, max_cloud_percentage[0])
        self.download_path = None
        self.downloaded_files = None
        self.products = None
        self.api = SentinelAPI(user_name, password, 'https://scihub.copernicus.eu/dhus')
        # search by polygon
        self.footprint = geojson_to_wkt(read_geojson(os.path.join(base_path, "poly.geojson")))
        # for searching by time
        self.start_date = start_date
        self.end_date = end_date
        self.tile_id = tile_id
        self.max_wind_speed = max_wind_speed

    # query api for matching products
    def get_product_data(self):
        # check to make sure wind is < max_wind_speed

            query_kwargs = {
                'platformname': 'Sentinel-2',
                'producttype': 'S2MSI1C',
                'date': (self.start_date, self.end_date)}

            kw = query_kwargs.copy()
            kw['cloudcoverpercentage'] = self.cloud_percentage
            if self.tile_id:
                self.products = OrderedDict()
                for tile in self.tile_id:
                    kw['tileid'] = tile
                    product = self.api.query(self.footprint, **kw)
                    if product:
                        # only keep product if wind_speed less than maximum (if wind speed filtered)
                        if check_wind(self.max_wind_speed, self.start_date):
                            # check size of product, ignore products smaller than 200mb (stops wasting time on small scenes
                            size = list(product.items())[0][1].get("size")
                            print(f"sentinel image size: {size}")
                            # remove unit (MB) convert to float
                            if float(size[:-2]) > 200:
                                self.products.update(product)
            else:
                self.products = self.api.query(self.footprint, **kw)
            # check for duplicate tiles (these can cause errors when merging)
            # this code keeps the newest product (by generation date)
            product_dict = {}
            self_products_copy = self.products.copy()
            for product, value in self.products.items():
                tile_id = value["title"].split("_")[-2]
                # product discriminator should be the same for any tiles with multiple datatakes, excluding the last 6
                # numbers indicating time of generation product
                product_discriminator = value["title"].split("_")[-1].replace("T", "")
                datetime_object = datetime.strptime(product_discriminator, '%Y%m%d%H%M%S')

                # don't download products less than 200MB in size
                size = value.get("size")
                # remove unit (MB) convert to float
                if float(size[:-2]) < 200:
                    del self_products_copy[product]

                try:
                    # delete oldest product of the tile
                    if datetime_object != product_dict[tile_id]:
                        del self_products_copy[product]

                except KeyError:
                    product_dict[tile_id] = datetime_object
            self.products = self_products_copy

            # *** this code creates a grid of ocean points for the poly.geojson file.
            # It checks these against the polygon of the sentinel-2 product and removes it from the download list
            # if the product does not contain any ocean that intersects with this area. ***
            # get the min and max latitudes and longitudes for the poly.geojson file
            if self.products:
                print("checking products contain ocean...")
                min_lon, max_lon, min_lat, max_lat = get_min_max_long_lat()
                # create an incremental range of points from these
                longitudes = np.arange(min_lon, max_lon, 0.01)
                latitudes = np.arange(min_lat, max_lat, 0.01)
                # check if these coords are ocean or not
                ocean_coords = []
                for lat in latitudes:
                    for lon in longitudes:
                        if globe.is_ocean(lat, lon):
                            # lon lat format for polygon intersection
                            ocean_coords.append([lon, lat])
                ocean_products = self.products.copy()
                # for each product, check to see if there is ocean within it (that also intersects with poly.geojson)
                for key, val in self.products.items():
                    polygon = shapely.wkt.loads(val['footprint'])
                    ocean_points = MultiPoint(ocean_coords)
                    product_ocean_points = polygon.intersection(ocean_points)
                    if product_ocean_points.is_empty:
                        del ocean_products[key]
                        print(f"no ocean detected in {key}, removing from analysis ")
                # final products
                self.products = ocean_products

    # download all products
    # consider a path_filter for some bands?
    def download(self,  directory_path):
        self.api.download_all(self.products, directory_path)

    def get_download_list(self):
        self.downloaded_files = [f for f in os.listdir(self.download_path) if os.path.isfile(os.path.join(self.download_path, f))]

    def run(self):
        self.get_product_data()
        if not self.products:
            info_string = "No products found for " + self.start_date + " with max cloud cover percentage of " + str(self.cloud_percentage[1])
            if self.max_wind_speed:
                info_string = info_string + " and max wind speed of " + self.max_wind_speed[0]
            print(info_string)
        self.download_path = os.path.join(base_path, "data", "downloads")
        self.download(directory_path=self.download_path)
        self.get_download_list()
        unzip_files(self.downloaded_files, self.download_path)
