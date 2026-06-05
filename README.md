# Stage-opdracht-2de-jaar
Voor school heb ik in de tweede jaar stage gelopen. Deze heb ik gedaan bij de Data Science Centre of Excellence, Defensie.
Daarvoor heb ik als opdracht om te kijken tot hoeverre ik een fuzzing ouput dataset kan gebruiken met AI. Deze repo is voor de bestanden die ik heb gebruikt te laten zien

**Note:** Het bestand `requests_responses.csv` en `transformed_data_sampled.csv` is beschikbaar in de [releases](https://github.com/WapixelXD/Stage-opdracht-2de-jaar/releases) vanwege de grootte van het bestand (> 100MB).

In deze repository zitten deze bestanden als volgt:
- Bestanden voor de corpus
- bestanden voor de data analyse
- Document met rapport
- De model bestand, code en datasets


In de readme.md staat als volgt:
- Link naar de tutorial van WuppieFuzz van TNO
- Directory tree, over hoe het repo is opgebouwd
- Pipeline script
- Korte uitleg over hoe je het project gebruikt.





## Directory Tree

Dit is hoe het project is ingedeeld:

```text
Stage-opdracht-2de-jaar/
├── .gitignore
├── Pipeline_Start.sh
├── README.md
├── fuzzing-project/
│   ├── header.yaml
│   ├── login.yaml
│   └── openapi.json
├── scripts/
│   ├── best_api_xgboost_model.json
│   ├── Confusion_Matrix.png
│   ├── EDA.ipynb
│   ├── Extract.py
│   ├── jacocoagent.jar
│   ├── Transform.py
│   └── XGBoost_Model.ipynb
|   └── XGBoost_Model.py
├── seeds/
└── requirements.txt
