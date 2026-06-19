# Stage-opdracht 2de Jaar - AI op Fuzzing Output

Dit project is uitgevoerd in het kader van een tweedejaars HBO-stage bij het Data Science Centre of Excellence (DSCE) van het Ministerie van Defensie. Het doel van het onderzoek is om te bepalen in hoeverre een dataset bestaande uit fuzzing-output geanalyseerd, verwerkt en geclassificeerd kan worden met behulp van kunstmatige intelligentie (AI), specifiek met een XGBoost-model.

## Projectomschrijving

Tijdens deze stage is onderzocht hoe machine learning kan worden toegepast op de resultaten van een fuzzing-campagne. De focus ligt hierbij op het analyseren van de gegenereerde netwerkverzoeken en de bijbehorende responsen van het doelsysteem. Er is een pijplijn ontwikkeld die deze data extraheert, opschoont en transformeert. Vervolgens herkent een XGBoost-model patronen in de fuzzing-output. De preprocessing-fase zorgt voor de noodzakelijke data-transformaties en de correcte verdeling van de doelvariabele, zodat het model effectief helpt bij het identificeren van afwijkend gedrag of potentiële kwetsbaarheden.

## Projectstructuur

De repository is als volgt opgebouwd:

```text
Stage-opdracht-2de-jaar/
├── .gitignore                  - Bestanden en mappen die genegeerd worden door Git.
├── Pipeline_Start.sh           - Shell-script om de volledige data- en modelpijplijn te starten.
├── README.md                   - Dit document.
├── requirements.txt            - De benodigde Python-packages voor dit project.
├── Onderzoeksverslag.pdf       - Het volledige theoretische en praktische onderzoeksrapport.
└── fuzzing-project/            - Alle bestanden en mappen gerelateerd aan het fuzzing-proces.
    ├── header.yaml             - HTTP-headers voor de fuzzing-verzoeken.
    ├── login.yaml              - Authenticatiegegevens voor de doelapplicatie.
    ├── openapi.json            - De API-specificatie van het geteste doelsysteem.
    ├── scripts/                - Scripts voor dataverwerking en modellering.
    │   ├── Extract.py          - Script voor het extraheren van de ruwe fuzzing-data.
    │   ├── Transform.py        - Script voor data-preprocessing, transformatie en feature engineering.
    │   ├── XGBoost_Model.py    - Python-script voor het trainen en evalueren van het XGBoost-model.
    │   ├── XGBoost_Model.ipynb - Jupyter Notebook voor experimenten en model-evaluatie.
    │   ├── EDA.ipynb           - Jupyter Notebook voor Exploratory Data Analysis.
    │   ├── best_api_xgboost_model.json - Het getrainde en opgeslagen XGBoost-model in JSON-formaat.
    │   └── Confusion_Matrix.png - Visuele weergave van de modelprestaties.
    └── seeds/                  - Inputbestanden (seeds) gebruikt om het fuzzing-proces te voeden.
```

## Setup

Vanwege de bestandsgrootte (> 100 MB) zijn de hoofd-datasets niet direct in de Git-geschiedenis opgenomen. Deze kunnen worden gedownload via de GitHub Releases-pagina:

* `requests_responses.csv`: De ruwe dataset met alle HTTP-verzoeken en responsen gegenereerd door de fuzzer.
* `transformed_data_sampled.csv`: De gesamplede en gepreprocesste dataset die direct wordt gebruikt voor het trainen van het XGBoost-model. De doelvariabele `status` is binnen de preprocessing-stap reeds gestratificeerd voor een evenwichtige klassenverdeling.

## Installatie en Vereisten

Om dit project lokaal uit te voeren, dienen de juiste Python-pakketten geïnstalleerd te zijn. Het wordt aanbevolen om een virtuele omgeving te gebruiken.

1. Kloon de repository:
```bash
git clone https://github.com/WapixelXD/Stage-opdracht-2de-jaar.git
cd Stage-opdracht-2de-jaar

```


2. Installeer de vereiste dependencies:
```bash
pip install -r requirements.txt

```


3. Download de benodigde datasets van de Releases-pagina en plaats deze in /fuzzing-projects/scripts.

## Gebruik van het Pipeline

De volledige pijplijn van data-extractie tot en met de model-evaluatie kan geautomatiseerd worden uitgevoerd met behulp van het meegeleverde shell-script. Zorg ervoor dat het script uitvoerbaar is:

```bash
chmod +x Pipeline_Start.sh
./Pipeline_Start.sh

```

Dit script doorloopt sequentieel de volgende stappen:

1. **Extractie (`fuzzing-project/scripts/Extract.py`)**: Het inlezen en verzamelen van de ruwe netwerkkorpus en logs uit de fuzzing-omgeving.
2. **Transformatie (`fuzzing-project/scripts/Transform.py`)**: Het opschonen van de data, het uitvoeren van feature engineering en het voorbereiden van de variabelen.
3. **Modellering (`fuzzing-project//XGBoost_Model.py`)**: Het trainen van het XGBoost-model op de getransformeerde dataset, het opslaan van het getrainde (`best_api_xgboost_model.json`) en het genereren van de Confusion Matrix (`Confusion_Matrix.png`).

*(Voor het geval als u uw eigen dataset gebruikt kunt u de bashscript aan passen zodat voor u werkt. Bijvoorbeeld: Transform-Extract-Modellering)*


## Resultaten en EDA
Voor  het EDA en visualiseringen kan de Jupyter Notebook (`EDA.ipynb`) worden geopend.
Het resultaat plaatje kunt vinden in zowel het `Onderzoeksverslag.pdf` en `Confusion_Matrix.png` vinden.





## Gebruik van het Opgeslagen Model
Als je enkel het getrainde model wilt gebruiken om voorspellingen te doen op nieuwe (getransformeerde) data, hoef je niet de hele trainingspijplijn opnieuw te doorlopen. Je kunt het JSON-model direct inladen via de Python-bibliotheek xgboost.

Hier is een voorbeeld van hoe je het model inlaadt en toepast:
```python
import xgboost as xgb
import pandas as pd

# 1. Initialiseer een lege XGBoost Classifier
model = xgb.XGBClassifier()

# 2. Laad het getrainde model in vanuit het JSON-bestand
model.load_model('fuzzing-project/scripts/best_api_xgboost_model.json')

# 3. Bereid je nieuwe of testdata voor (zorg dat de features exact overeenkomen met de training)
# nieuwe_data = pd.read_csv('jouw_nieuwe_data.csv')

# 4. Doe voorspellingen
# voorspellingen = model.predict(nieuwe_data)
# print(voorspellingen)
```



## Verwijzingen

* **TNO WuppieFuzz**: Dit project maakt gebruik van of is geïnspireerd door de methodologie en tutorials van WuppieFuzz, ontwikkeld door TNO. Raadpleeg de officiële TNO-documentatie voor verdere achtergrondinformatie over deze fuzzing-architectuur.

**Voor het geval als u zelf WuppieFuzz wilt gebruiken om eigen data te creeëren wordt het aangeraden om naar https://github.com/TNO-S3/WuppieFuzz en vervolgens naar de tutorial map te gaan om WuppieFuzz in te stellen.**
