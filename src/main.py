import geopandas as gpd
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import requests
import rasterio
from rasterio.transform import from_origin
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import numpy as np
import time
import os
import csv
from io import BytesIO
import textwrap
import configparser
from pyproj import Transformer
import random as rnd

def authenticate(client_id, client_secret):
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    token = oauth.fetch_token(
        token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        client_secret=client_secret,
        include_client_id=True
    )
    if (token is not None) or (token!="") or (token["access_token"] is not None):
        return {"token":token["access_token"], "expires_in":token["expires_in"]}
    else:
        return -1
    
def get_area(path):
    # Lettura file KML
    gdf = gpd.read_file(path, driver="KML")

    # utilizzo il primo poligono
    polygon = gdf.geometry[0]

    # coordinate in formato [(lon, lat, alt)]
    polygon_coords = list(polygon.exterior.coords)
    return polygon_coords

def make_request(token, width, height, start, end, polygon_coords, maxCloudCoverage=30):
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "input": {
            "bounds": {"geometry": {"type": "Polygon", "coordinates": [polygon_coords]}},
            "data": [{
                "dataFilter": {"timeRange": {"from":start,"to":end}, "maxCloudCoverage": maxCloudCoverage},
                "type":"sentinel-2-l2a"
            }]
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier":"default","format":{"type":"image/tiff"}}]
        },
        "evalscript": """//VERSION=3
        function setup() {
            return {
                input: ["B03","B04","B05","B08","B11","SCL","dataMask"],
                output:[
                    {id:"default", bands:7, sampleType: "FLOAT32" },
                    {id:"scl", bands:1, sampleType:"INT8"},
                    {id:"dataMask", bands:1, sampleType: "INT8" }
                ]
            };
        }

        function evaluatePixel(sample) {
            let B03 = sample.B03 / 10000.0; // Green
            let B04 = sample.B04 / 10000.0; // Red
            let B05 = sample.B05 / 10000.0; // Red Edge 1
            let B08 = sample.B08 / 10000.0; // NIR
            let B11 = sample.B11 / 10000.0; // SWIR

            // Indici vegetativi
            let ndvi = (B08 - B04) / (B08 + B04);
            let ndre = (B08 - B05) / (B08 + B05);
            let ndmi = (B08 - B11) / (B08 + B11);
            let gci  = (B08 / B03) - 1.0;

            return {
                default: [ndvi, ndre, ndmi, gci, B08, B04, B05],
                scl: [sample.SCL],
                dataMask: [sample.dataMask]
            };
        }"""
    }
    return {"url": url, "headers": headers, "data": data}


def process_req(req, start, end, filename): # salvataggio csv dei dati grezzi
    # lettura del tif
    empty=False
    xs=[]
    ys=[]
    response = requests.post(req["url"], headers=req["headers"], json=req["data"])
    if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("image/"):
        bin=BytesIO(response.content)
        with rasterio.open(bin) as src:
            arr = src.read()
            empty=(np.all(np.isnan(arr) | (arr == 0)))

            if not empty:
                transform = src.transform

                # Calcola le coordinate X,Y reali
                rows, cols = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing='ij')
                xs, ys = rasterio.transform.xy(transform, rows, cols)
                xs = np.array(xs)
                ys = np.array(ys)

                # conversione pixel in df
                df = pd.DataFrame({
                    "X": xs.flatten(),
                    "Y": ys.flatten(),
                    "NDVI": arr[0].flatten(),
                    "NDRE": arr[1].flatten(),
                    "NDMI": arr[2].flatten(),
                    "GCI":  arr[3].flatten(),
                    "B08":  arr[4].flatten(),
                    "B04":  arr[5].flatten(),
                    "B05":  arr[6].flatten(),
                })

                df.to_csv(filename, index=False)
    else:
        print("ERRORE DI COMUNICAZIONE")
        print(response.text)

    return filename

def view_image(path):
    band_info = {
        "NDVI": "Densità  e vigoria delle piante. Valori buoni: 0.3-0.8 per vegetazione sana, <0 indica acqua, ~0 terreno nudo.",
        "NDRE": "Indice di clorofilla (vegetazione sana o meno). Valori buoni: <0.4 stress, >0.6 sano e maturazione frutti.",
        "NDMI": "Indice di umidità del suolo e densità vegetazione. Valori buoni: -0.4 - 0.4 (stress idrico), <0.8 ok. indica vegetazione umida, valori negativi terreno secco o stress idrico.",
        "GCI": "Clorofilla superficiale. Valori buoni: >2 indicano buona clorofilla, valori bassi stress o foglie giovani/ingiallite.",
        #"B08": "Banda NIR (Near Infrared, riflettanza del fogliame). Valori buoni: 0.3-0.7 per vegetazione sana, vicino a 0 per acqua o terreno nudo.",
        #"B04": "Banda Red (rosso, riflettanza foglie). Valori buoni: 0.05-0.2 vegetazione sana, più alto indica stress o terreno scoperto.",
        #"B05": "Banda Red Edge 1 (transizione rosso-NIR, stress vegetativo). Valori buoni: 0.05-0.25 per vegetazione sana, valori bassi stress o scarsità di fogliame."
    }
    interpretazioni = "**Interpretazioni delle combinazioni dei 4 indici:**\n\n\nA. Vegetazione sana e ben irrigata → NDVI, NDRE, NDMI, GCI alti.\n\nB. Vegetazione fitta ma povera di clorofilla → NDVI e NDRE alti, GCI basso.\n\nC. Vegetazione rada ma foglie sane → NDVI medio, NDMI e GCI alti.\n\nD. Vegetazione densa ma stress idrico → NDVI e NDRE alti, NDMI basso.\n\nE. Vegetazione scarsa/stress elevato → Tutti bassi.\n\nF. Suolo nudo o post-raccolta → NDVI, NDRE, NDMI molto bassi, GCI irrilevante.\n"

    # Apri il TIFF
    with rasterio.open(path) as src:
        arr = src.read()[:4]
        n_bands = arr.shape[0]

        n_rows = (n_bands + 1) // 2
        n_cols = 3  # due colonne per grafici + 1 per testo

        fig = plt.figure(figsize=(14, 5 * n_rows))

        gs = fig.add_gridspec(n_rows, n_cols, width_ratios=[2.8, 2.8, 1.8], wspace=0.4, hspace=0.6)

        for i in range(n_bands):
            row = i // 2
            col = i % 2
            ax = fig.add_subplot(gs[row, col])

            band_name = list(band_info.keys())[i]
            description = band_info[band_name]
            # Wrapping più stretto per i titoli e font più piccolo
            wrapped_desc = textwrap.fill(description, width=30)

            im = ax.imshow(arr[i], cmap='viridis', interpolation="bicubic")
            ax.set_title(f"{band_name}:\n{wrapped_desc}", fontsize=7, loc='left')
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, orientation='horizontal')

        ax_text = fig.add_subplot(gs[:, 2])
        ax_text.axis('off')

        wrapped = textwrap.fill(interpretazioni, width=50)
        ax_text.text(0, 1, wrapped, fontsize=10, va='top', ha='left', multialignment='left')

        # Margini più larghi per evitare fuoriuscite
        plt.subplots_adjust(left=0, right=0.95, top=0.9, bottom=0.05)

        plt.show()


def mosaic(fn, images):
    bande = ["NDVI", "NDRE", "NDMI", "GCI"]
    profile = {
        "driver": "GTiff",
        "height": zoom_height,
        "width": zoom_width,
        "count": len(bande),
        "dtype": "float32",
        "crs": "+proj=latlong",  # sistema di riferimento spaziale
        "nodata": np.nan
    }

    for b in range(len(bande)):
        raster_stack={} # accumulo dati di ogni banda
        minX=maxY=pixel_size_x=pixel_size_y=None
        for img in images:
            with open(img) as raw:
                reader=csv.DictReader(raw)

                for row in reader:
                    if (row["X"], row["Y"]) not in raster_stack:
                        if row[bande[b]]=="":
                            row[bande[b]]=np.nan

                        raster_stack[(row["Y"], row["X"])]=[float(row[bande[b]])]
                    else:
                        raster_stack[(row["Y"], row["X"])].append(float(row[bande[b]]))

            if minX==None:
                df=pd.read_csv(img)
                minX=df["X"].min()
                maxY=df["Y"].max()
                pixel_size_x=df["X"].diff().min()
                pixel_size_y=df["Y"].diff().abs().min()
                transform = from_origin(minX, maxY, pixel_size_x, pixel_size_y)
                profile["transform"] = transform

        for px in raster_stack:
            raster_stack[px]=np.average(raster_stack[px]) # media di un px per quella banda

        sorted_coords = sorted(raster_stack.keys(), key=lambda k: (-float(k[0]), float(k[1])))
        data = np.array([raster_stack[coord] for coord in sorted_coords]).reshape((int(zoom_height), int(zoom_width)))


        mode="r+"
        if b==0: mode="w"
        
        with rasterio.open(fn, mode, **profile) as img:
            img.write(data,b+1)


        raster_stack={}

def get_size(polygon_coords):
    # Calcolo width, height
    lon=[float(c[0]) for c in polygon_coords]
    lat=[float(c[1]) for c in polygon_coords]

    # calcolo del sistema di destinazione di coordinate
    mean_lon=np.average(lon)
    mean_lat=np.average(lat)

    zone=int((mean_lon+180)/6)+1 # identifico in quale frazione di zona mi trovo
    epsg_dest=32600+zone if mean_lat>=0 else 32700+zone

    # trasformatore da gradi (GPS - EPSG:4326) a metri
    transformer=Transformer.from_crs("EPSG:4326", epsg_dest, always_xy=True)
    xs,ys=transformer.transform(lon,lat) # trasformazione da gradi a metri

    zoom_width=max(xs)-min(xs)
    zoom_height=max(ys)-min(ys) 

    return {"zoom_width":zoom_width, "zoom_height":zoom_height}

parser = argparse.ArgumentParser()
parser.add_argument("-start")
parser.add_argument("-end")
parser.add_argument("-area")
parser.add_argument("-show")
parser.add_argument("-ow")


args=parser.parse_args()
start=args.start
end=args.end
filename_area=args.area


show=True
ow=True
if args.show=="False": show=False
if args.ow=="False": ow=False


config=configparser.ConfigParser()
config.read("../conf/conf.ini")

CLIENT_ID=config["AUTH"]["CLIENT_ID"]
CLIENT_SECRET=config["AUTH"]["CLIENT_SECRET"]

REQ_RATE=int(config["REQ"]["MINUTE_RATE"])


# Autenticazione
auth=authenticate(CLIENT_ID, CLIENT_SECRET)
token=auth["token"]
expires_in=auth["expires_in"]-(auth["expires_in"]*0.5)

polygon_coords=get_area(filename_area)

sizes=get_size(polygon_coords)

zoom_width=sizes["zoom_width"]
zoom_height=sizes["zoom_height"]


start_date = datetime.fromisoformat(start.replace("Z", ""))
end_date   = datetime.fromisoformat(end.replace("Z", ""))

images=[]
current = start_date
req_count=0

start_batch_time=time.time()
exp=time.time()+(auth["expires_in"]*0.5)
while current <= end_date:
    day_start = current.isoformat() + "Z"
    day_end = (current + relativedelta(days=1)-relativedelta(seconds=1)).isoformat() + "Z"

    now=time.time()
    fn=f"../data/{os.path.basename(filename_area).split('.')[0]}_{day_start.replace(':','-')}_{day_end.replace(':','-')}_pixels.csv" # pattern filename csv

    if ow or (not os.path.exists(fn)):

        # controllo che non supero il rate di req/min
        if (now-start_batch_time)>=60:
            start_batch_time=now
            req_count=0

        if (req_count==(REQ_RATE-1)):
            print("IN ATTESA PER EVITARE SUPERAMENTO RATE RICHIESTE/MINUTO")
            time.sleep(now-start_batch_time+1)
            start_batch_time=now
            req_count=0

        # verifico che non sia scaduto il token
        if now>=exp:
            auth=authenticate(CLIENT_ID, CLIENT_SECRET)
            token=auth["token"]
            exp=now+(auth["expires_in"]*0.5)

        req = make_request(
            token,
            width=zoom_width,
            height=zoom_height,
            start=day_start,
            end=day_end,
            polygon_coords=polygon_coords,
            maxCloudCoverage=31
        )

        fn=process_req(req, day_start, day_end, fn)
        if not os.path.exists(fn): fn=None

        if fn is not None:
            print(f"Intervallo {day_start} - {day_end}")
        else:
            print(f"Intervallo {day_start} - {day_end}: NESSUNA IMMAGINE DISPONIBILE")

        req_count+=1

    if show:
        if (fn is not None) and os.path.exists(fn):
            images.append(fn)

    current += relativedelta(days=1)


# mosaicking delle immagini
if show:
    if images:
        mosaic_filename=f"../data/out{rnd.random()}.tif"
        mosaic(mosaic_filename, images)
        view_image(mosaic_filename)
        os.remove(mosaic_filename)