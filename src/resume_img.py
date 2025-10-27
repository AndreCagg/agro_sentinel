from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import os
import argparse


def describe_indices(values):
    # Dizionario di regole per ogni indice
    rules = {
        "NDVI": [
            (0.0, "area non vegetata o terreno zuppo"),
            (0.3, "scarsa densità vegetativa"),
            (0.6, "moderata densità vegetativa"),
            (1.0, "zona densamente vegetata")
        ],
        "NDRE": [
            (0.2, "vegetazione altamente stressata ed improduttiva"),
            (0.4, "vegetazione moderatamente stressata produttivamente"),
            (0.6, "vegetazione produttiva"),
            (1.0, "vegetazione sana e vigorosa")
        ],
        "GCI": [
            (1, "Scarsa qta clorofilla - vegetazione inattiva"),
            (2, "Poca qta clorofilla - vegetazione poco attiva"),
            (3, "Sufficiente clorofilla - vegetazione attiva"),
            (4, "Molta clorofilla - vegetazione molto attiva"),
            (10, "Abbondante clorofilla - vegetazione abbondantemente attiva")
        ],
        "NDMI": [
            (0.0, "area non vegetata"),
            (0.1, "vegetazione con elevato stress idrico (o in spigamento)"),
            (0.4, "scarsa idratazione"),
            (0.6, "vegetazione regolare con esigenze idriche"),
            (1.0, "vegetazione con riserve idriche abbondanti")
        ]
    }

    description_parts = []

    for index_name, value in values.items():
        if index_name in ["NDRE"]: continue
        thresholds = rules.get(index_name)
        if not thresholds:
            continue  # salta indici non riconosciuti

        for threshold, text in thresholds:
            if value <= threshold:
                description_parts.append(text)
                break


    efficiency = float(values["GCI"] / values["NDRE"])

    if efficiency < 6:
        message = f"{efficiency:.2f} - Vegetazione giovane o potata (0-6)"
    elif 6 <= efficiency <= 10:
        message = f"{efficiency:.2f} - Vegetazione in salute (6 - 10)"
    elif 10 < efficiency < 12:
        message = f"{efficiency:.2f} - Vegetazione leggermente stressata, attenzionare eventuali patogeni (10-12)"
    elif 12 <= efficiency < 15:
        message = f"{efficiency:.2f} - Vegetazione inefficiente, clorofilla inattiva, monitorare per patologie(12 - 15)"
    else:  # efficiency >= 15
        message = f"{efficiency:.2f} - Vegetazione molto malata (>15) - patologie"


    description_parts.append("\n\nClorofilla disponibile su clorofilla biochimicamente attiva GCI/NDRE: "+message) # quanto della clorofilla presente è biochimicamente attiva
    # Composizione automatica della frase
    return ", ".join(description_parts)

# AGGREGAZIONE
parser = argparse.ArgumentParser()
parser.add_argument("-start")
parser.add_argument("-end")
parser.add_argument("-area")


args=parser.parse_args()
start=args.start
end=args.end

filename_area=os.path.basename(args.area).split(".")[0]

start_date = datetime.fromisoformat(start.replace("Z", ""))
end_date   = datetime.fromisoformat(end.replace("Z", ""))

current = start_date

bands={"NDVI":[], "NDRE":[], "NDMI":[], "GCI":[]}

# valutazione su immagine aggregata
while current <= end_date:
    day_start = current.isoformat() + "Z"
    day_end   = (current + relativedelta(days=1)-relativedelta(seconds=1)).isoformat() + "Z"
    year = current.year
    month = current.month
    
    filename=f"../data/{filename_area}_{day_start.replace(':','-')}_{day_end.replace(':','-')}_pixels.csv"

    if os.path.exists(filename):
        df=pd.read_csv(filename)

        df=df[(df["NDVI"]!=0) & (df["NDRE"]!=0) & (df["NDMI"]!=0) & (df["GCI"]!=0)]

        ndvi=df["NDVI"].mean()
        ndre=df["NDRE"].mean()
        ndmi=df["NDMI"].mean()
        gci=df["GCI"].mean()

        bands["NDVI"].append(float(ndvi))
        bands["NDRE"].append(float(ndre))
        bands["NDMI"].append(float(ndmi))
        bands["GCI"].append(float(gci))


    current+=relativedelta(days=1)

ndvi=np.average(bands["NDVI"])
ndre=np.average(bands["NDRE"])
ndmi=np.average(bands["NDMI"])
gci=np.average(bands["GCI"])

print(f"NDVI: {ndvi}")
print(f"NDRE: {ndre}")
print(f"NDMI: {ndmi}")
print(f"GCI: {gci}")

values={"NDVI": ndvi, "NDRE": ndre, "NDMI": ndmi, "GCI": gci}

print("\n\n"+describe_indices(values))