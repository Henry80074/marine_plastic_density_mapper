
from rasterio.merge import merge
import rasterio
import numpy as np
import os
import sys

from dotenv import load_dotenv

print(sys.path)
sys.path.append("..")
from coordinate_generators import generate_threshold_coords, generate_plastic_coordinates, generate_plastic_coordinates2
from mean_pixel_values import latlon2distance, my_mean
from utils.dir_management import get_files, base_path
from pyproj import Transformer
import plotly.figure_factory as ff
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rasterio import merge
from data_filters import mean_pixel_value_filter, bathymetry_filter, class_percentages_filter, port_mask


load_dotenv()


mapbox_token = os.environ.get('MAPBOX')
# plotly_params
hex_bin_number = 6  # higher number results in smaller and more numerous hex_bins
hex_bin_opacity = 0.5  # 0 (invisible) to 1 (opaque)
data_marker_size = 2  # plotted plastic detection marker size
data_marker_opacity = 0.4  # 0 (invisible) to 1 (opaque)
min_detection_count = 0  # lowest number of plastic detections needed for a hex_bin to be visualised on the map
max_plastic_percentage = 0.1 # remove all dates
close_to_land_mask = 2 # mask detections close to land (in km)
land_blur = 0.008999*close_to_land_mask
#excessive fmasking indicates scene is highly impacted by clouds or sunglint, removes dates with high mask percentage
max_masking_percentage = 80
min_depth = 5
port_mask_distance = 2500

# code to find and plot all suspected plastic on a map
def plot_data(df):
    if not df.empty:
        fig = ff.create_hexbin_mapbox(
            data_frame=df, lat="latitude", lon="longitude",
            nx_hexagon=hex_bin_number, opacity=hex_bin_opacity, labels={"color": "Point Count"},
            show_original_data=True,
            original_data_marker=dict(size=data_marker_size, opacity=data_marker_opacity, color="deeppink"),
            color_continuous_scale="Reds", min_count=min_detection_count,

        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(b=0, t=0, l=0, r=0))
        fig.show()


def plot_data_by_day(df):
    dates = pd.unique(df['date'])
    for date in dates:
        filtered_df = df.loc[(df['date'] == date)]
        if not df.empty:
            fig = ff.create_hexbin_mapbox(
                data_frame=filtered_df, lat="latitude", lon="longitude",
                nx_hexagon=hex_bin_number, opacity=hex_bin_opacity, labels={"color": "Point Count"},
                show_original_data=True,
                original_data_marker=dict(size=data_marker_size, opacity=data_marker_opacity, color="deeppink"),
                color_continuous_scale="Reds", min_count=min_detection_count,
                title="Plastic Detections for " + date
            )
            fig.update_layout(mapbox_style="open-street-map", margin=dict(b=0, t=40, l=0, r=0))
            fig.show()


# use to threshold probabilities file
def plot_probabilities(tag, data_path, threshold):
    all_point_data = []
    for file in sorted(get_files(data_path, tag)):
        all_point_data.extend(generate_threshold_coords(file, threshold))

    df = pd.DataFrame(all_point_data, columns =['latitude', 'longitude'])
    if not df.empty:
        fig = ff.create_hexbin_mapbox(
            data_frame=df, lat="latitude", lon="longitude",
            nx_hexagon=hex_bin_number, opacity=hex_bin_opacity, labels={"color": "Point Count"},
            show_original_data=True,
            original_data_marker=dict(size=data_marker_size, opacity=data_marker_opacity, color="deeppink"),
            color_continuous_scale="Reds", min_count=min_detection_count
        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(b=0, t=0, l=0, r=0))
        fig.show()


def save_coordinates_to_csv(tiff_path, tag):
    all_point_data = []
    for file in sorted(get_files(tiff_path, tag)):
        date = os.path.basename(file).split("_")[1]
        tile = os.path.basename(file).split("_")[1]
        all_point_data.extend(generate_plastic_coordinates2(file, date, tile))
    df = pd.DataFrame(all_point_data, columns=['latitude', 'longitude', 'date', 'tile_id', 'plastic_percentage', 'mask_percentage'])
    df = df.drop_duplicates()

    if os.path.exists(os.path.join(base_path, "data", "outputs", "plastic_coordinates.csv")):
        df.to_csv(os.path.join(base_path, "data", "outputs", "plastic_coordinates.csv"), mode="a", header=False)
    else:
        df.to_csv(os.path.join(base_path, "data", "outputs", "plastic_coordinates.csv"), mode="w", header=True)


# plots
def plot_data_single_day(date):
    df = pd.read_csv(os.path.join(base_path, "data", "outputs", "plastic_coordinates.csv"))
    if not df.empty:
        is_date = df['date'] == int(date)
        df = df[is_date]
        fig = ff.create_hexbin_mapbox(
            title="Plastic Detections for " + date,
            data_frame=df, lat="latitude", lon="longitude",
            nx_hexagon=hex_bin_number, opacity=hex_bin_opacity, labels={"color": "Point Count"},
            show_original_data=True,
            original_data_marker=dict(size=data_marker_size, opacity=data_marker_opacity, color="deeppink"),
            color_continuous_scale="Reds", min_count=min_detection_count,
        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(b=0, t=40, l=0, r=0))
        fig.show()
    else:
        print("no detections for " + date + " skipping plot...")


def plot_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    fig = ff.create_hexbin_mapbox(
        title="Plastic Detections for ",
        data_frame=df, lat="latitude", lon="longitude",
        nx_hexagon=hex_bin_number, opacity=hex_bin_opacity, labels={"color": "Point Count"},
        show_original_data=True,
        original_data_marker=dict(size=data_marker_size, opacity=data_marker_opacity, color="deeppink"),
        color_continuous_scale="Reds", min_count=min_detection_count,
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(b=0, t=40, l=0, r=0))
    fig.show()


# code related to calculating MDM metric values
def get_pixel_mean_values(probabilities_tag, data_path, fname, filtered_df):
    '''
    calculating mdm values for each pixels and returning detected MD&SP pixels in a csv file with their corresponding MDM values
    '''
    dates = pd.unique(filtered_df["date"])
    temp_files = []
    for file in sorted(get_files(data_path, probabilities_tag)):
        date = os.path.basename(file).split("_")[1]
        # don't process dates that have been removed from analysis already
        if date not in dates:
            pass
        # create temporary files for calculating mean values
        else:
            src = rasterio.open(file)
            meta = src.meta
            image = src.read(1)
            image[image == 99] = 0
            image[image > 1] = 0
            image[np.isnan(image)] = 0
            pure_probability = file.strip("probabilities_masked.tif") + "probability_layer.tif"
            with rasterio.open(pure_probability, "w", **meta) as dst:
                dst.write(image,indexes=1)
            temp_files.append(pure_probability)
            image[image > 0.99] = 1
            image[image != 1] = 0
            pure_detections = file.strip("probabilities_masked.tif") + "prediction_layer.tif"
            with rasterio.open(pure_detections, "w", **meta) as dst:
                dst.write(image, indexes=1)
            temp_files.append(pure_detections)

    # sum pixel probabilities across all dates
    probability_files_to_mosaic = []
    for file in sorted(get_files(data_path, "probability_layer")):
        src = rasterio.open(file)
        probability_files_to_mosaic.append(src)
    prob_mosaic, out_trans = merge.merge(probability_files_to_mosaic, method='sum', nodata=0)
    predictions_files_to_mosaic = []
    for file in sorted(get_files(data_path, "prediction_layer")):
        src = rasterio.open(file)
        predictions_files_to_mosaic.append(src)
    pred_mosaic, out_trans = merge.merge(predictions_files_to_mosaic, method='sum', nodata=0)

    # remove temporary files
    for file in temp_files:
        if file.endswith("prediction_layer.tif") or file.endswith("probability_layer.tif"):
            os.remove(file)

    # calculate mean pixel values
    I = prob_mosaic[0] / (len(temp_files)/2)  # taking average of probability maps
    I3 = I * (100 * pred_mosaic[0] / (len(temp_files)/2))  # MDM calculation in pixel level
    detect_pixels = np.argwhere(I3 > 0.01)  # getting rid of higly small MDM values
    pix = len(detect_pixels)

    # create dataframe with coordinates and pixel values
    geo_coords = [rasterio.transform.xy(meta['transform'], coord[0], coord[1], offset='center') for coord in
                  detect_pixels]
    transformer = Transformer.from_crs(src.crs, "epsg:4326")
    coords = [transformer.transform(coord[0], coord[1]) for coord in geo_coords]
    vals = []
    for c, i in enumerate(detect_pixels):
        vals.append(I3[i[0], i[1]])
        print("Processed {}/{}".format(c, pix))
    all_point_data2 = []
    for i in range(len(coords)):
        dated_coord = list(coords[i])
        dated_coord.append(vals[i])
        all_point_data2.append(dated_coord)
    df3 = pd.DataFrame(all_point_data2, columns=['latitude', 'longitude', 'vals'])
    
    # only return df values for filtered coordinates
    df = filtered_df.drop(['date'], axis=1)
    df4 = df.merge(df3, on=["latitude", "longitude"])
    df4 = df4.dropna()
    df4 = df4.drop_duplicates()
    fname3 = fname + ".csv"
    df4.to_csv(os.path.join(base_path, "data", "outputs", fname3), index=False)
    return df4


# code related to MDM metric visualisation
def plot_mean_pixel_val_map(file):
    df3 = pd.read_csv(file)
    c1 = [np.abs(df3.latitude.mean()), np.abs(df3.longitude.max())]
    c2 = [np.abs(df3.latitude.mean()), np.abs(df3.longitude.min())]
    hexagon_size = np.round(latlon2distance(c1, c2) / 5).astype(int)
    df4 = df3.nlargest(50, 'vals')  # taking 10 largest MDM values
    if not df3.empty:
        px.set_mapbox_access_token(mapbox_token)
        fig = ff.create_hexbin_mapbox(
            data_frame=df3, lat="latitude", lon="longitude", color='vals', agg_func=my_mean,
            nx_hexagon=hexagon_size, opacity=0.5, labels={"color": "MPM"},
            original_data_marker=dict(size=3, opacity=0.0, color="crimson"),
            show_original_data=True, range_color=(0, 4),
            # setting max to 1.5. this is manual but for better visualisation
            color_continuous_scale="Greys", min_count=1)
        fig.add_trace(go.Scattermapbox(  # this is for adding maximum MDM values
            lat=df4.latitude,
            lon=df4.longitude,
            mode='markers', showlegend=False,
            marker=go.scattermapbox.Marker(size=8, color=df4.vals, colorbar=dict(orientation='h', yanchor="bottom",
                                                                                 y=-0.15, xanchor="center", title='MDM',
                                                                                 x=0.5, len=0.9), opacity=0.99, cmin=0,
                                           cmax=17, colorscale='Rainbow')))
        fig.update_layout(mapbox_style="streets", width=700, height=600)
        fig.update_layout(coloraxis_colorbar=dict(title="50%<br> Trimmed<br> Mean<br> MDM"))
        fig.show()


# applies various filters to reduce false positives
def filter_data(data_path, prediction_tag, bathymetry=True, MDM=True, ports=True):

    # TODO [WIP]
    # if bathymetry:
    #     src_files_to_mosaic = []
    #     dir = os.path.join(base_path, "utils", "bathymetry_maps", "gebco_2023_tid_geotiff")
    #     tiff_files = [x for x in os.listdir(dir) if x.endswith(".tif")]
    #     for fp in tiff_files:
    #         src = rasterio.open(os.path.join(dir, fp))
    #         src_files_to_mosaic.append(src)
    #     out_meta = src.meta.copy()
    #     mosaic, out_trans = merge(src_files_to_mosaic)
    #
    #     out_meta.update({"driver": "GTiff",
    #                      "height": mosaic.shape[1],
    #                      "width": mosaic.shape[2],
    #                      "transform": out_trans,
    #                      "crs": "EPSG:4326"})
    #
    #     with rasterio.open(os.path.join(base_path, "utils", "bathymetry_maps", "merged_bath_map.tiff"), "w", **out_meta) as dest:
    #         dest.write(mosaic)

    fname = os.path.basename(data_path)

    # TODO - remove need for data path and prediction_tag for calulating scene plastic percentages
    #  this has now been refactored and is calculated during pipeline runs so should be obtained from plastic_coordinates.csv
    # filters pixels if based on excessive plastic detections, excessive masking or too close to land
    df = class_percentages_filter(data_path, prediction_tag, max_plastic_percent=max_plastic_percentage, max_masking_percent=max_masking_percentage, land_blurring=land_blur)

    # TODO implement bathymetery filter
    # filter pixels in too shallow water (1km resolution)
    # if not df.empty and bathymetry:
    #     df = bathymetry_filter(df, min_depth=min_depth, file=bathymetry_file)

    # filter pixels with two high MDM (static detections may be caused by other land or ocean features)
    if not df.empty and MDM:
        df = get_pixel_mean_values("probabilities_masked", data_path, fname, df)
    if not df.empty and MDM:
        df = mean_pixel_value_filter(df)
    # filter pixels close to ports and anchorages (ships and other vessels are sometimes misclassified)
    if not df.empty and ports:
        df = port_mask(df, port_mask_distance)
    plot_data(df)
    df.to_csv(os.path.join(base_path, "data", "outputs", "plastic_coordinates_filtered.csv", index=False))
    return df


if __name__ == "__main__":
    print("Analysing MAP-Mapper outputs and plotting plastic detections...")
    plot_data_from_csv(os.path.join(base_path, "data", "outputs", "plastic_coordinates.csv"))

