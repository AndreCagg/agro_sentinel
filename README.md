

# Agro Sentinel



Questo programma è ancora un proof-of-concept, ovvero la verifica se gli strumenti utilizzati ed utilizzabili sono sufficienti per raggiungere l'obiettivo.



L'obiettivo è quello di realizzare un software per l'identificazione precoce di potenziali malattie agronomiche tramite le rilevazioni satellitari di Sentinel-2.



Al momento è stato implementato solamente l'aspetto della raccolta e/o visualizzazione delle rilevazioni in un certo periodo senza determinare potenziali patogeni in fase di sviluppo. Il sistema attuale permette solamente una valutazione, da parte dell'operatore, dello stress idrico e vegetativo delle colture.



I dati utilizzati dal sistema sono file ```.csv```.











## Installation



Per l'installazione è sufficiente il download del repository e l'esecuzione del ```main.py```

```bash

python ./main.py

```

## Configuration


Nel file ```conf/conf.ini``` vanno personalizzati i parametri di ```CLIENT_ID```, ```CLIENT_SECRET``` E ```MINUTE_RATE```. Il ```MINUTE_RATE``` esprime il numero massimo di richieste che Sentinel-Hub accetta in un minuto.


## Usage/Examples


```bash

python ./main.py

```



i parametri da inserire sono

```-start```: data di inizio periodo del quale si ha interesse in formato ```yyyy-mm-ddThh:mm:ssZ```


```-end```: come per ```start```


```-area:```: path per il file kml contenente il poligono dell'area di interesse


```-show: True | False```: per la visualizzazione dell'immagine aggregata con le medie del periodo specificato


```-ow: True | False```: overwrite, se sono disponibili in locale già i dati desiderati e vanno sovrascritti o meno


Nella cartella ```asset``` è presente il file ```.kml``` della zona di interesse. AL momento è possibile utilizzare solo un poligono e non un multipoligono



Nella cartella ```data``` sono presenti i file ```.csv``` scaricati che rappresentano i pixel dell'imamgine.

Durante il download di un range di valori può capitare di ricevere un errore dal server (5xx) indicante il superamento del rate di richieste al minuto, questo perchè in una esecuzione precedente sono state già fatte altre richieste che il sistema, nell'esecuzione corrente, non può tracciare.



# License

Questo software è realizzato sotto la ```MIT License```



