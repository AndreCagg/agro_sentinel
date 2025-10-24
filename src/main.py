import geopandas as gpd
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import requests
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import numpy as np
import textwrap
import time
import os

def authenticate(client_id, client_secret):
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    token = oauth.fetch_token(
        token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        client_secret=client_secret,
        include_client_id=True
    )
    if (token is not None) or (token!="") or (token["access_token"] is not None):
        #token = token["access_token"]
        return {"token":token["access_token"], "expires_in":token["expires_in"]}
    else:
        return -1
    
def get_area(path):
    # Lettura file KML
    gdf = gpd.read_file(path, driver="KML")

    # utilizzo il primo poligono
    polygon = gdf.geometry[0]

    # coordinate in formato [lon, lat]
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
            "width": width, #10m
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


def process_req(req, start, end, csv): # download immagine e salvataggio csv
    filename_tif=f"../data/{start.replace(':','-')}_{end.replace(':','-')}.tif"
    # scarica TIFF
    response = requests.post(req["url"], headers=req["headers"], json=req["data"])
    if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("image/"):
        with open(filename_tif, "wb") as f:
            f.write(response.content)

    else:
        print("ERRORE")
        print(response.text)

    # lettura del tif
    empty=False
    if response.status_code==200:
        with rasterio.open(filename_tif) as src:
            arr = src.read()
            empty=(np.all(np.isnan(arr) | (arr == 0)))
            """if not empty:
                print(arr)"""
    else:
        print("ERRORE DI CONNESSIONE")

    #valid_mask = np.isfinite(arr).all(axis=0) & (arr[4] != 0) & (arr[5] != 0) & (arr[6] != 0) # se l'immagine è vuota
    #if np.any(valid_mask):
    #if empty: print("VUOTAAAAA")
    if not empty:
        if csv:
            # conversione pixel in df
            df = pd.DataFrame({
                "NDVI": arr[0].flatten(),
                "NDRE": arr[1].flatten(),
                "NDMI": arr[2].flatten(),
                "GCI":  arr[3].flatten(),
                "B08":  arr[4].flatten(),
                "B04":  arr[5].flatten(),
                "B05":  arr[6].flatten(),
            })

            #df=df[(df["B08"]!=0) & (df["B04"]!=0) & (df["B05"]!=0)]
            df = df[~((df["B08"] == 0) & (df["B04"] == 0) & (df["B05"] == 0))] # elimina righe tutte 0

            filename_csv=f"../data/{start.replace(':','-')}_{end.replace(':','-')}_pixels.csv"
            

            df.to_csv(filename_csv, index=False)

            #print("CSV salvato con", len(df), "pixel.")
            """
            if show:
                view_image(filename_tif)"""
    else:
        print("NESSUNA IMMAGINE DISPONIBILE")
        os.remove(filename_tif)
        filename_tif=None
        filename_csv=None

    return filename_tif

def view_image(path):
    band_info = {
        "NDVI": "Indice di vegetazione (densità delle piante). Valori buoni: 0.5-0.8 per vegetazione sana, <0 indica acqua, ~0 terreno nudo.",
        "NDRE": "Indice di vegetazione Red Edge (stress della vegetazione). Valori buoni: 0.3-0.6 indicano fogliame sano, valori bassi indicano stress.",
        "NDMI": "Indice di umidità del suolo e vegetazione. Valori buoni: -0.2 - 0.4 (poco stress idrico), <0.8 ok. indica vegetazione umida, valori negativi terreno secco o stress idrico.",
        "GCI": "Contenuto di clorofilla nella vegetazione. Valori buoni: 2-4 indicano buona clorofilla, valori bassi stress o foglie giovani/ingiallite.",
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

            im = ax.imshow(arr[i], cmap='viridis')
            ax.set_title(f"{band_name}:\n{wrapped_desc}", fontsize=7, loc='left')
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, orientation='horizontal')

        ax_text = fig.add_subplot(gs[:, 2])
        ax_text.axis('off')

        #wrapped_interpretazioni = textwrap.fill(interpretazioni, width=40)
        #fig.text(0.05, 0.5, interpretazioni, fontsize=9, va='center', ha='left', multialignment='left')
        #ax_text.text(0, 1, wrapped_interpretazioni, fontsize=10, va='top', ha='left', family='monospace')
        wrapped = textwrap.fill(interpretazioni, width=50)
        ax_text.text(0, 1, wrapped, fontsize=10, va='top', ha='left', multialignment='left')

        # Margini più larghi per evitare fuoriuscite
        plt.subplots_adjust(left=0, right=0.95, top=0.9, bottom=0.05)

        plt.show()




parser = argparse.ArgumentParser()
parser.add_argument("-csv")
parser.add_argument("-start")
parser.add_argument("-end")
parser.add_argument("-show")

#start="2018-01-01T00:00:00Z"
#end="2025-10-22T23:59:59Z"

args=parser.parse_args()
start=args.start
end=args.end


csv=False
show=True
if args.csv=="True": csv=True
if args.show=="False": show=False


CLIENT_ID="sh-fb5d1f38-1268-4edf-8189-5205a88e76cf"
CLIENT_SECRET="KV6K6YiIHmM9gnudWGKJbgKyrIokMjP1"

# Autenticazione
auth=authenticate(CLIENT_ID, CLIENT_SECRET)
token=auth["token"]
expires_in=auth["expires_in"]-(auth["expires_in"]*0.5)

polygon_coords=get_area("../asset/uliveto.kml")

"""polygon_coords = [
    [16.551187,40.650726],[16.551125,40.650523],[16.55212,40.650385],
    [16.552155,40.650409],[16.552187,40.650442],[16.552209,40.65048],
    [16.552222,40.650519],[16.552233,40.650549],[16.552187,40.650592],
    [16.552093,40.650653],[16.552002,40.650724],[16.551935,40.650765],
    [16.551843,40.65069],[16.551763,40.650663],[16.551575,40.650692],
    [16.551411,40.650722],[16.551187,40.650726]
]"""

#start="2018-01-01T00:00:00Z"
#end="2025-10-22T23:59:59Z"
start_date = datetime.fromisoformat(start.replace("Z", ""))
end_date   = datetime.fromisoformat(end.replace("Z", ""))

#if batch:
images=[]
current = start_date
count=0

exp=time.time()+(auth["expires_in"]*0.5)
while current <= end_date:
    if time.time()>=exp:
        auth=authenticate(CLIENT_ID, CLIENT_SECRET)
        token=auth["token"]
        exp=time.time()+(auth["expires_in"]*0.5)

    day_start = current.isoformat() + "Z"
    day_end = (current + relativedelta(days=1)-relativedelta(seconds=1)).isoformat() + "Z"
    print(f"Intervallo {day_start} - {day_end}")

    req = make_request(
        token,
        width=38.8198125985208,
        height=60.413087653464956,
        start=day_start,
        end=day_end,
        polygon_coords=polygon_coords,
        maxCloudCoverage=31
    )

    fn=process_req(req, day_start, day_end, csv)

    current += relativedelta(days=1)
    count+=1
    if show:
        if fn is not None:
            images.append(fn)

    if count==20:
        time.sleep(20)
        count=0

# mosaicking delle immagini
if show:
    bande = ["NDVI", "NDRE", "NDMI", "GCI"]
    mosaics=[]
    for b in range(len(bande)):
        raster_stack=[] # accumulo dati di ogni banda
        for img in images:
            with rasterio.open(img) as r:
                raster_stack.append(r.read(b+1)) # prendo la corrispondente banda, l'ordine è tassativamente questo (vedi nella richiesta)
            
        raster_stack = np.stack(raster_stack, axis=0) # creo array 3D (n_immagini, altezza, larghezza)
        mean_band = np.mean(raster_stack, axis=0) # fa la media

        mean_band = mean_band.astype(raster_stack[0].dtype) # la media della banda
        mosaics.append(mean_band) # si crea il mosaico

    # profilo raster
    with rasterio.open(images[0]) as src0:
        profile = src0.profile
        profile.update(count=len(bande))  # aggiorna numero di bande per l'output

    # realizzazione del mosaico multi-banda
    output_path = "../data/mosaic_output.tif"
    with rasterio.open(output_path, "w", **profile) as dst:
        for idx, band in enumerate(mosaics, start=1):
            dst.write(band, idx)

    view_image(output_path)