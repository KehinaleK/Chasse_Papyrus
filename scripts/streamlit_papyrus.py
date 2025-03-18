import streamlit as st
import pandas as pd
from annotated_text import annotated_text
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

### Voici le script permettant d'obtenir notre petite application streamlit ! 
### Pour ce qui est de la fonctionnalité supplémentaire, j'ai rajouté deux modes
### qui permettent d'obtenir des chiffres au sujet des papyri au sein d'un même intervalle
### de dates ou qui proviennent du même lieu.

### DE PLUS ! Je sais qu'il fallait originellement seulement trois scripts python. Mais j'ai voulu faire en sorte d'avoir les 
### lieux sur une carte. Pour ça, j'avais fait une fonction pour récupérer les fichiers json pendant le scrapping mais étant 
### donné que nous repartions de votre tableau pour corpus_analyse, j'ai refais du scrap à partir de votre colonne Places_List
### pour obtenir les latitutdes et longitudes de la page geo-ref. Le scrapping pour obtenir tout ça est un peu long (au moins 15min chez moi !).
### Pour éviter d'avoir à scrapper des pages à chaque lancement de l'application, j'ai fait ce scrapping en amont et ai stocké les coordonnées trouvées
### dans un fichier 'coordonnées.csv' stocké avec les autres tables ! J'ai aussi laissé le script 'scrap_geo.py' dans le dossier de scripts
### mais il n'y a donc pas besoin de le lancer pour obtenir les cartes, c'est déjà fait. Merci !!!

def also_in(df: pd.DataFrame, label: str, column: str, value: set, data: pd.Series) -> None:

    """Cette fonction répèrtorie les papyri dans lesquels
    se trouve l'entité sélectionnée."""

    ### On crée un expander pour lister les lieux et personnes parce que je trouve ça very very nice
    with st.expander(f"{label} dans le texte"):
        for entity in value:
            rows = df[df[column].str.contains(entity)]
            rows = rows[rows["ID"] != data["ID"].values[0]]
            ids = rows["ID"].to_list()
            if st.button(entity):
                if len(ids) != 0:
                    st.write(f"Apparaît aussi dans {', '.join(map(str, ids))}.")
    

def load_csv(csv_file: str) -> pd.DataFrame:

    """Cette fonction permet de charger nos données et de les convertir en dataframe."""

    df = pd.read_csv(csv_file)
    return df


def process_data(df: pd.DataFrame) -> tuple[list[int], int, int]:

    """Cette fonction permet d'adapter nos données. Nous récupérons notamment les plus vieilles et récentes
    dates et nettoyons la colonne provenance pour retirer les ponctuations. (et éviter les doublons)"""

    ### On garde que les premiers mots comme pour dans corpus analysis ! 
    all_dates = [int(date) for intervalle in df['Date Intervalles'] for date in intervalle.strip("(").strip(")").split(",")]
    min_date = min(all_dates)
    max_date = max(all_dates)
    df["Provenance"] = df["Provenance"].apply(lambda x: x.strip("-").strip("?"))

    return all_dates, min_date, max_date

def display_home_page(mode: str) -> None:

    """Cette fonction affiche les informations présentes initialement au chargement de la page."""

    ### Le mode accueil est le mode de base, il permet d'afficher le titre et la description seulement au début ! 
    if mode == "Accueil":
        st.markdown("""
            <h1 style='text-align: center; color: white;'>📜 La Chasse aux Papyrus 📜</h1>
            <h3 style='text-align: center; color: white; font-style: italic;'>Explorez les trésors antiques de la collection 
                    <a href='https://www.trismegistos.org/'>Trismegistos</a></h3>

        <div style='margin-top: 20px; padding: 16px;'>
            <p style='color: white; text-align: justify; font-size: 16px; font-'>
                Cette application vous permet d'explorer le monde des papyri, des supports d'écriture antiques principalement issus du nord-est de l'Afrique et de la Grèce. Découvrez les précieux documents de cette collection à travers trois modes :
            </p>
            <ul style='color: white; font-size: 16px;'>
                <li><strong>Recherche de papyrus :</strong> Choisissez un papyrus pour accéder à des informations détaillées</li>
                <li><strong>Chiffres par lieu :</strong> Explorez les données concernant les papyri provenant d'un lieu spécifique</li>
                <li><strong>Chiffres par date :</strong> Obtenez des informations sur les papyri compris dans un intervalle temporel</li>
            </ul>
            <p style='color: white; font-style: italic; text-align: center;'>Bon voyage ! Ἀπόλαυσις !</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
        """
        <style>
            [data-testid=stImage]{
                text-align: center;
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 100%;
            }
        </style>
        """, unsafe_allow_html=True
        )
        st.image("../data/img/front.jpg")



def choose_mode() -> str:

    """Cette fonction gère la sélection du mode choisi par l'utilisateur."""
    
    st.markdown("""
    <style>
    [data-testid="stSidebarContent"] {
        color: black;
        background-color: #457b9d;
    }
    </style>
    """, unsafe_allow_html=True)
    mode = st.sidebar.selectbox("", options=["Accueil", "Recherche de papyrus","Chiffres par lieu", "Chiffres par date"], format_func=lambda x: "Accueil" if x == "Accueil" else x)
    return mode


def filter_papyri(df: pd.DataFrame, all_dates: list[int], min_date: int, max_date: int) -> str:

    """Cette fonction permet de gérer la sélection d'un papyrus dans le mode
    'recherche de papyrus' incluant les filtres."""

    ### J'ai utilisé les session state pour updater le slider en fonction du lieu choisi ! Mais j'ai pas réussi à faire l'inverse car de ce
    ### que j'ai compris, les selectbox ont des choix statiques. Donc on ne peut pas les updater en fonction des dates choisies par exemple ou alors
    ### il faudrait imposer un ordre de filtres... Ou alors j'ai juste pas réussi hehe (surement)

    if "date_range" not in st.session_state:
            st.session_state["date_range"] = (min_date, max_date)

    ### SelectBox pour les lieux
    filtrage_lieu = st.sidebar.selectbox("Trier par lieu", options=df["Provenance"].unique(), placeholder="Choisir un lieu", index=None)

    ### Update des papyris de la selectBox papyrus et du slider
    if filtrage_lieu is not None:
        df = df[df["Provenance"] == filtrage_lieu]
        all_dates = [int(date) for intervalle in df['Date Intervalles'] for date in intervalle.strip("(").strip(")").split(",")]
        min_date = min(all_dates)
        max_date = max(all_dates)
        st.session_state["date_range"] = (min_date, max_date)

    ### Slider pour les dates (adapté aux lieux choisis)
    filtrage_date = st.sidebar.slider(label="Trier par date", min_value=min_date, max_value=max_date, value=(min_date, max_date))

    ### Update des papyris de la selectBox
    if filtrage_date is not None:
        df = df[df["Date Intervalles"].apply(lambda x: filtrage_date[0] <= int(x.strip("(").strip(")").split(",")[0]) <= filtrage_date[1] and filtrage_date[0] <= int(x.strip("(").strip(")").split(",")[1]) <= filtrage_date[1])]

    ### SelectBox pour les papyri
    papyrus = st.sidebar.selectbox("Choisissez un papyrus", options=df["ID"], placeholder="Choisir un papyrus", index=None)

    return papyrus

def display_papyrus_infos(df: pd.DataFrame, papyrus: str) -> None:

        """Cette fonction permet de gérer l'affichage des informations pertinentes
        concernant le papyrus choisi."""

        ### On récupère les données de la ligne du papyrus
        data = df.loc[df["ID"] == papyrus]

        st.header(f"Papyrus {papyrus}")
     
        labels = ["Date", "Provenance", "Personnes", "Lieux", "Texte"]
        date = data["Date"].values[0] if not pd.isnull(data["Date"].values[0]) else "Unknown"
        provenance = data["Provenance"].values[0] if not pd.isnull(data["Provenance"].values[0]) else "Unknown"
        if data["People List Clean"].values[0] != "[]":
            people = data["People List Clean"].values[0].strip("[").strip("]").replace("'", "")
            people = set(people.split(","))
        else:
            people = "Aucun individu n'a pu être identifié"

        if data["Places List Clean"].values[0] != "[]":
            places = data["Places List Clean"].values[0].strip("[").strip("]").replace("'", "")
            places = set(places.split(","))
            df_filtered_places = map_dataframe(places)
            if df_filtered_places.empty:
                exist = False
            else:
                exist = True
        else:
            places = "Aucun lieu n'a pu être identifié"
            exist = False
        
        if data["Text Irregularities"].values[0] != "[]":
            irregularities = data["Text Irregularities"].values[0].strip("[").strip("]").replace("'", "")
            irregularities = set(irregularities.split(","))
        else:
            irregularities = "Aucune irrégularité n'a pu être identifiée"

        values = [date, provenance, people, places, irregularities]

        ### On crée nos petites colonnes miam miam
        for label, value in zip(labels, values):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{label}**")
            with col2:
                if "Aucun" in value:
                    st.write(f"_{value}_")
                else:
                    if label == "Date" or label == "Provenance":
                        st.write(value)
                    else:
                        if label == "Personnes":
                            also_in(df, label, "People List Clean", value, data)
                        elif label == "Lieux":
                            also_in(df, label, "Places List Clean", value, data)

        correspondances = {}           
        text = data["Full Clean"].values[0]
        for pair in irregularities:
            pair = pair.split(":")
            if len(pair) == 2: 
                old = pair[1].replace("read", "").strip()
                new = pair[0].strip()
                if old != "" and new != "":
                    correspondances[new] = old

        text_with_annotations = []

        for word in text.split():
            if word.strip("*") in correspondances.keys():
                text_with_annotations.append((word.strip("*"), correspondances[word.strip("*")]))
            else:
                text_with_annotations.append(word)
            text_with_annotations.append(" ")

        annotated_text(text_with_annotations)

        ### Si on a des lieux alors hop hop on check dans les coordonnées pour créer la map
        ### Mais en véririant plusieurs papyri, je n'ai qu'un lieu à chaque fois sur la carte.
        ### Je ne sais pas si le problème vient de mauvaises correspondances (même avec nettoyage)
        ### ou si c'est parce que généralement, c'est toujours le même lieu qui est mentionné dans un texte
        if exist:
            st.map(df_filtered_places, latitude="Lat", longitude="Lon", size=10, color=None)


def filter_dates(df: pd.DataFrame, dates: tuple[int, int]) -> pd.DataFrame:
    
    """Cette fonction permet de filtrer la dataframe en fonction de l'intervalle de dates choisi."""

    df = df[df["Date Intervalles"].apply(lambda x: dates[0] <= int(x.strip("(").strip(")").split(",")[0]) <= dates[1] and dates[0] <= int(x.strip("(").strip(")").split(",")[1]) <= dates[1])]
    return df

def filter_places(df: pd.DataFrame, location: str) -> pd.DataFrame:
    
    """Cette fonction permet de filtrer la dataframe en fonction du lieu choisi."""
    
    df = df[df["Provenance"] == location]
    return df

def display_places_numbers(location: str, df_filtered: pd.DataFrame) -> None:
    
    """Cette fonction gère la génération et l'affichage des informations du mode `chiffres par lieu'."""

    num_papyris = len(df_filtered)
    st.header(location)

    st.subheader(f"La collection contient {num_papyris} documents(s) pour la ville de {location}.")

    st.subheader(f"Les dates couvertes par les papyri sont les suivantes :")
    all_date_points = []
    date_intervals = df_filtered["Date Intervalles"].to_list()
  
    for interval in date_intervals:
        interval = interval.split(",")
        points = np.arange(int(interval[0].strip("'").strip("(")), int(interval[1].strip("'").strip(")")), 1).tolist()
        all_date_points.extend(points)

    years, counts = np.unique(all_date_points, return_counts=True)

    fig, ax = plt.subplots()
    ax.plot(years, counts, color='b', label='Nombre de papyrus') 
    ax.fill_between(years, counts, color='b', alpha=0.3) 
    ax.set_title("Intervalles de dates de parution")
    ax.set_xlabel("Année")
    ax.set_ylabel("Nombre de papyrus")
    ax.legend()

    st.pyplot(fig)

def display_dates_numbers(dates: tuple[int, int], df: pd.DataFrame) -> None:
    
    """Cette fonction gère la génération et l'affichage des informations du mode `chiffres par date'."""
    
    df["Provenance"] = df["Provenance"].str.split().str[0]
    location_count = df["Provenance"].value_counts().sort_values(ascending=False)
    
    fig, ax = plt.subplots()
    ax.set_title(f"Répartition des papyrus en fonction du lieu de provenance - Intervalle {dates}")
    location_count.plot(kind='bar', ax=ax, figsize=(10, 6)) 
    
    st.pyplot(fig)

def most_common(location: str, df: pd.DataFrame) -> None:
    
    """Cette fonction permet d'afficher les entités les plus citées dans les modes 'chiffres'."""

    all_places = []
    all_people = []
    for idx, row in df.iterrows():
        people = row["People List Clean"]
        places = row["Places List Clean"]
        if people != "[]":
            people = people.strip("[").strip("]").replace("'", "").split(",")
            all_people.extend(people)
        if places != "[]":
            places = places.strip("[").strip("]").replace("'", "").split(",")
            all_places.extend(places)

    if all_people != []:
        count_people = Counter(all_people)
        df_count_people = pd.DataFrame.from_dict(count_people, orient="index").reset_index()
        df_count_people = df_count_people.rename(columns={'index':'Personne', 0:'Occurrences'})
        df_count_people = df_count_people.sort_values(by='Occurrences', ascending=False)
        st.subheader(f"Les {len(df_count_people.head(10))} individus les plus cités dans les papyri de {location} :")
        st.table(df_count_people.head(10))
    
    if all_places != []:
        count_places = Counter(all_places)
        df_count_places = pd.DataFrame.from_dict(count_places, orient="index").reset_index()
        df_count_places = df_count_places.rename(columns={'index':'Lieu', 0:'Occurrences'})
        df_count_places = df_count_places.sort_values(by='Occurrences', ascending=False)
        st.subheader(f"Les {len(df_count_places.head(10))} endroits les plus cités dans les papyri de {location} :")
        st.table(df_count_places.head(10))


def map_dataframe(places: set) -> pd.DataFrame:
    
    """Cette fonction permet d'afficher les lieux citées sur une carte."""

    df = pd.read_csv("../data/tables/coordonnees.csv")
    df_filtered = df[df["Lieu"].isin(places)]

    return df_filtered

def main():

    mode = choose_mode()
    display_home_page(mode)
    
    if mode == "Recherche de papyrus":
        df = load_csv("../data/tables/clean_papyrus-corpus.csv") ### on ré-actualise à chaque changement de mode
        st.title("Recherche de Papyrus")
        all_dates, min_date, max_date = process_data(df)
        papyrus = filter_papyri(df, all_dates, min_date, max_date)
        if papyrus:
            display_papyrus_infos(df, papyrus)


    elif mode == "Chiffres par lieu":
        df = load_csv("../data/tables/clean_papyrus-corpus.csv")
        st.title("Chiffres par Lieu")
        all_dates, min_date, max_date = process_data(df)
        location = st.sidebar.selectbox("Trier par lieu", options=df["Provenance"].unique(), placeholder="Choisir un lieu", index=None)
        df_filtered = filter_places(df, location)
        if location:
            display_places_numbers(location, df_filtered)
            most_common(location, df_filtered)


    elif mode == "Chiffres par date":
        df = load_csv("../data/tables/clean_papyrus-corpus.csv")
        st.title("Chiffres par Date")
        all_dates, min_date, max_date = process_data(df)
        dates = st.sidebar.slider(label="Trier par date", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        df_filtered = filter_dates(df, dates)
        if df_filtered.empty:
            st.subheader(f"_Aucun document pour l'intervalle {dates}_")
        else : 
            display_dates_numbers(dates, df_filtered)
            most_common(dates, df_filtered)


if __name__ == "__main__":
    main()