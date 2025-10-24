

\# Agro Sentinel



Questo programma è ancora un proof-of-concept, ovvero la verifica se gli strumenti utilizzati ed utilizzabili sono sufficienti per raggiungere l'obiettivo.



L'obiettivo è quello di realizzare un software per l'identificazione precoce di potenziali malattie agronomiche tramite le rilevazioni satellitari di Sentinel-2.



Al momento è stato implementato solamente l'aspetto della raccolta e/o visualizzazione delle rilevazioni in un certo periodo senza determinare potenziali patogeni in fase di sviluppo. Il sistema attuale permette solamente una valutazione, da parte dell'operatore, dello stress idrico e vegetativo delle colture.



I dati utilizzati dal sistema sono ```.tif``` e ```.csv``` ma si punta ad utilizzare solo i ```.csv```.











\## Installation



Per l'installazione è sufficiente il download del repository e l'esecuzione del ```main.py```

```bash

&nbsp; python ./main.py

```

&nbsp;   

\## Usage/Examples



```bash

python ./main.py

```



i parametri da inserire sono



```-csv: True | False```: per la creazione dei file ```.csv``` oltre che dei ```.tif```. I file ```.csv``` rappresentano l'immagine non georeferenziata (prossimamente verranno eliminati i file ```.tif``` aggiungendo la georeferenziazione nei ```.csv```)



```-start```: data di inizio periodo del quale si ha interesse in formato ```yyyy-mm-ddThh:mm:ssZ```



```-end```: come per ```start```



```-show: True | False```: per la visualizzazione dell'immagine aggregata con le medie del periodo specificato



Nella cartella ```asset``` è presente il file ```.kml``` della zona di interesse, il path del file è ancora hard-coded nel main.



Nella cartella ```data``` sono presenti i file ```.csv``` e ```.tif``` scaricati



\# License

Questo software è realizzato sotto la ```MIT License```



