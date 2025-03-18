import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import numpy as np
import argparse
import unicodedata as ud 
from difflib import ndiff
from typing import Tuple, List, Dict
from collections import defaultdict, Counter
from pyvis.network import Network

"""Bonjour, voici le deuxième script 'corpus_analysis' du TP 2 'La Chasse au Papyrus'.
Le script contient un argparse pour que vous puissiez obtenir les réponses aux parties désirées.
Celà empêche d'attendre 5 minutes pour l'execution de la partie sur les entités nommées (︶︹︶).
Pour pouvoir obtenir le fichier csv utilisé pour la partie streamlit, veuillez utiliser 
l'argument 'streamlit' défini plus bas ! J'effectue quelques modifications et nettoyages
pour faciliter la création de l'application comme indiqué dans la partie 'transition' du sujet.
Nous ne sommes aussi pas obligés d'avoir la partie sur les NER pour obtenir le csv donc 
économie de temps ! (et d'énergie)

Exemple pour obtenir les réponses d'une partie :
    python3 corpus_analysis.py -P 4
    [Vous pouvez choisir parmi 4, 5, 6, 7 et 8, streamlit]

Exemple pour obtenir le csv `clean_papyrus-corpus.csv` :
    python3 corpus_analysis.py -P streamlit


Les tableaux sont stockés dans data/tables et le graph dans data/graph !
    
Bonne correction !"""


def read_data(csv_file: str) -> pd.DataFrame:
    """Cette fonction permet de convertir nos données en dataframe."""

    df = pd.read_csv(csv_file)

    return df

def clean_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """Cette fonction permet de retirer les papyri sans texte."""

    len_before = len(df)

    bye_bye_non_text = df.loc[df["Full Text"].isnull()].index
    df = df.drop(bye_bye_non_text)

    len_after = len(df)

    how_many_null_text = len_before - len_after

    return df, how_many_null_text, len_after

def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cette fonction permet de trier la dataframe en fonction des ID des papyri."""

    df = df.sort_values(by=["ID"], key=lambda x:x.str[2:].astype(int))

    return df

def get_content(df: pd.DataFrame) -> None:

    """Permet de voir quel type de document se trouve dans la collection."""

    all_contents = df["Content (beta!)"].value_counts()
    # Pour voir que nous avons des petits parasites... see See : 

    df["Content (beta!)"] = df["Content (beta!)"].astype(str).str.lower().replace(to_replace=r":", value="", regex=True)
    df["Content (beta!)"] = df["Content (beta!)"].astype(str).str.lower().replace(to_replace=r"See", value="", regex=True)
    df["Content (beta!)"] = df["Content (beta!)"].astype(str).str.lower().replace(to_replace=r"see", value="", regex=True)
   
    df["Content (beta!)"] = df["Content (beta!)"].str.split().str[0]
    
    content_count = df.groupby(["Content (beta!)"]).size()

    labels = content_count.keys()
    perc = [(value / len(df)) * 100 for key, value in content_count.items()]

    plt.figure()
    plt.title("Pourcentage de type de documents !")
    plt.pie(perc, labels=labels, autopct="%1.1f%%", normalize=True)

    plt.show()
    plt.close()


def recycled_papyri(df: pd.DataFrame) -> int:

    """Cette fonction permet d'obtenir le nombre de papyri recyclés."""

    num_recycled_papyri = df["Reuse type"].count() + df["Reuse note"].count()
    
    return num_recycled_papyri

def get_provenance(df: pd.DataFrame) -> None:

    """Cette fonction permet de visualiser les villes de provenance des documents."""

    df["Provenance"] = df["Provenance"].str.split().str[0]

def visualise_provenance(df: pd.DataFrame) -> None:

    content_count = pd.Series(df.groupby(["Provenance"]).size().sort_values(ascending=False))

    plt.figure()
    plt.title("Répartition des papyrus en fonction du lieu de provenance.")
    content_count.plot.bar(figsize=(6,6))
    plt.show()

def formating_dates(df: pd.DataFrame) -> List[Tuple[int, int]]:

    """Cette fonction permet d'obtenir une colonne contenant les intervalles temporels de chaque papyrus.
    Exemple : 563 jun-oct 589 => 563 à 589"""

    all_dates = []
    for id, row in df.iterrows():
        value = row["Date"]
        matches = re.search(r'((AD|BC) [0-9]{3}( (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))?( [0-9]{1,2})?( - [0-9]{3})?)', value)
        dates = matches.group(0)
        all_dates.append(dates)

    ### On obtient un match qui conserve tous les intervalles, mais certains contiennent des mois et etc...
    ### Donc on va nettoyer ça !

    date_intervals = []
    for date in all_dates:
        matches_years = re.findall(r'[0-9]{3}', date)
        if matches_years:
            if len(matches_years) == 2:
                matches_years = (int(matches_years[0]), int(matches_years[1]))
            if len(matches_years) == 1:
                matches_years = (int(matches_years[0]), int(matches_years[0]) + 1)
                ### Pour compter un point quand on a une seule année !
                ### Si 566, on obtient 566-567
        
            date_intervals.append(matches_years)
    
    df["Date Intervalles"] = date_intervals

    return date_intervals

def density_maps_of_dates(date_intervals: List[Tuple[int, int]]) -> None:

    """Cette fonction permet de visualiser la densité de papyri en fonction des intervalles temporels."""

    all_date_points = []
    for interval in date_intervals:
        points = np.arange(interval[0], interval[1], 1).tolist()
        all_date_points.extend(points)

    sns.kdeplot(all_date_points, bw_adjust=0.5, fill=True)

    plt.title("Intervals historiques")
    plt.xlabel("Année")
    plt.ylabel("Densité")

    plt.show()

def first_clean_function(text: str) -> str:

    """Cette fonction est l'une des fonction permettant de nettoyer les textes.
    Elle retire les chiffres arabes des textes ainsi que les symboles † et ⳨."""

    matches_lost_lines = re.findall(r"\|gap=[0-9]*_lines\|", text)
    matches_arabic_numbers = re.findall(r"[0-9]+", text)
    if matches_lost_lines:
        for match in matches_lost_lines:
            text = text.replace(match, "")

    if matches_arabic_numbers:
        for match in matches_arabic_numbers:
            text = text.replace(match, "")

    text = text.replace("†", "").replace("⳨", "")

    return text

def second_clean_function(text: str) -> str:

    """Cette fonction est l'une des fonctions permettant de nettoyer le texte.
    Cette fonction permet de stocker la proportion de caractères incertain dans une nouvelle colonne."""

    ### Je ne suis pas sûre de bien avoir compris la notation des incertitudes mais j'ai admis que les caractères incertains sont ceux
    ### avec un point en dessous. Les portions incertaines sont entre crochets ou parenthèses et les portions illisibles sont les 
    ### tirets et points entre parenthèses ou entre crochets.

    matches_bad_portions = re.findall(r"\[[\u0370-\u03FF\u1F00-\u1FFF]+\]|\([\u0370-\u03FF\u1F00-\u1FFF]+\)", text)
    # Pour trouver les [] et ()
    matches_bad_chars = re.findall(r".\u0323", text)
    # Pour trouver les caractères avec points ! 

    bad_portions = [x.replace("[", "").replace("]", "").replace("(", "").replace(")", "") for x in matches_bad_portions]

    bad_portions = sum([len(x) for x in bad_portions]) if matches_bad_portions is not None else 0
    bad_chars = len(matches_bad_chars) if matches_bad_chars is not None else 0
    uncertain = bad_portions + bad_chars

    proportion = (uncertain / len(text)) * 100

    return proportion

def third_clean_function(text: str) -> str:

    """Cette fonction fait partie des fonctions permettant de nettoyer le texte.
    Cette fonction retire les paranthèses et crochets."""

    text = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")

    return text

def fourth_clean_function(text: str) -> str:

    """Cette fonction permet de nettoyer de nouveau le texte pour améliorer les prédictions du modèle UGARIT."""

    text = text.replace(".", "").replace("-", "").replace("/", "").replace("_", "")

    return text

def clean_text(df: pd.DataFrame) -> None:

    """Cette fonction permet de nettoyer le texte et stocke le résultat dans une colonne "Full Clean",
    les proportions d'incertitude sont stockées dans la colonne "Uncertain Portion"."""

    df["Full Clean"] = df["Full Text"].apply(first_clean_function)
    df["Uncertain Portion"] = df["Full Text"].apply(second_clean_function)
    df["Full Clean"] = df["Full Clean"].apply(third_clean_function)
    df["Full Clean"] = df["Full Clean"].apply(fourth_clean_function)

def how_many_too_uncertain_texts(df: pd.DataFrame) -> int:

    """Cette fonction permet d'obtenir le nombre de textes dont la proportion de caractères incertains est
    supérieure à un tiers."""

    uncertain_texts = len(df[df["Uncertain Portion"] > 33])

    return uncertain_texts

def column_clean_function_people(list: str) -> List[str]:
    
    """Cette fonction permet de nettoyer la colonne "People List" en ne gardant que les
    caractères grecs et en retirant les éventuels points ou chiffres."""
    
    list = list.strip("[").strip("]").replace("'", "").replace("", "").split(",")
    pretty_people = []
    for maybe_person in list:
        greek_letters = re.findall(r"[\u0370-\u03FF\u1F00-\u1FFF]+", maybe_person)
        if greek_letters:
            person = re.sub(r"[0-9]*|\.+", "", maybe_person)
            if person != "":
                person = person.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(".", "").replace("-", "").replace("/", "").replace("_", "")
                pretty_people.append(person.strip())

    return pretty_people

def column_clean_function_places(list: str) -> List[str]:
    
    """Cette fonction permet de nettoyer la colonne "Places List" en convertissant
    chaque valeur en liste de lieux et en retirant les numéros géo !"""

    list = list.strip("{").strip("}")
    if list == "":
        return []
    
    list = list.replace("'", "").replace(":", "").split(",")
    pretty_places = []
    for place in list:
        place = re.sub(r"[0-9]*", "", place)
        if place != "":
            place = place.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(".", "").replace("-", "").replace("/", "").replace("_", "")
            pretty_places.append(place.strip())

    return pretty_places

def get_entities(df: pd.DataFrame) -> None:

    """Cette fonction permet d'obtenir trois nouvelles colonnes contenant les personnes,
    lieux et autres entités nommées reconnues par le modèle UGARIT. Ces colonnes sont formatées
    de la même manière que les colonnes "Places List" et "People List"."""

    from transformers import pipeline

    ner = pipeline('ner', model="UGARIT/grc-ner-bert", aggregation_strategy = 'first')

    # On lance le modèle sur tout le texte du papyrus
    all_entites = []
    for id, row in df.iterrows():
        text = row["Full Clean"]
        entities = ner(text)
        all_entites.append(entities)
    
    ugarit_people = []
    ugarit_places = []
    ugarit_other = []

    for entity in all_entites:
        people = [group["word"] for group in entity if group["entity_group"] == "PER"]
        places = [group["word"] for group in entity if group["entity_group"] == "LOC"]
        others = [group["word"] for group in entity if group["entity_group"] == "MISC"]
        ugarit_people.append(people)
        ugarit_places.append(places)
        ugarit_other.append(others)

    df["People Ugarit"] = ugarit_people
    df["Places Ugarit"] = ugarit_places
    df["Other Ugarit"] = ugarit_other

def get_f1_score_severe(actual_values: List[List[str]], predicted_values: List[List[str]]) -> None:

    """Cette fonction permet d'obtenir un score de f1 sévère pour évaluer le modèle UGARIT.
    La fonction peut être lancée pour évalué la reconnaissance des lieux ou bien des personnes.
    Il suffit de donner la colonne originelle et la colonne UGARIT correspondante en arguments."""

    # Les valeurs ont été obtenues manuellement car les calculs de scickit learn par exemple, demandent
    # des structures différentes de celles que nous avons. Et puis on peut mieux verifier les valeurs ici.

    TP = 0
    FP = 0
    FN = 0

    # Les TP, FP et FN sont calculés sur l'ensemble des couples vraie valeur/prédiction. 
    # À chaque match, on retire les valeur concernées de chaque liste pour ne pas avoir 2 VP
    # dans ce cas par exemple : prédiction1 = actual_value, prediction1 = actual_value46.
    # Nous n'avons ici qu'un VP et un FN !

    all_actual_values = [value for sublist in actual_values for value in sublist]
    all_actual_values = [value.replace("[", "").replace("]", "").replace("(", "").replace(")", "") for value in all_actual_values]
    all_predicted_values = [value for sublist in predicted_values for value in sublist]

    remaining_actual_values = all_actual_values.copy()
    remaining_predicted_values = all_predicted_values.copy()

    # On normalise chaque paire et on lower le tout.

    for predicted_value in all_predicted_values:
        for actual_value in all_actual_values:
            normalised_predicted_value = ud.normalize("NFD", predicted_value.lower())
            normalised_predicted_value = ''.join(c for c in normalised_predicted_value if ud.category(c) != 'Mn')
            normalised_actual_value = ud.normalize("NFD", actual_value.lower())
            normalised_actual_value = ''.join(c for c in normalised_actual_value if ud.category(c) != 'Mn')
            if normalised_predicted_value == normalised_actual_value:
                TP += 1
                if predicted_value in remaining_predicted_values:
                    remaining_predicted_values.remove(predicted_value)
                if actual_value in remaining_actual_values:
                    remaining_actual_values.remove(actual_value)


    if len(remaining_actual_values) > 0:
        FN += len(remaining_actual_values)

    if len(remaining_predicted_values) > 0:
        FP += len(remaining_predicted_values)

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    f1_score = (2 * precision * recall) / (precision + recall)
    print(f"Le f1 score sévère est de {f1_score}.")

def get_f1_score_tolerant(actual_values: List[List[str]],predicted_values: List[List[str]],ner_predicted_values: List[List[str]]) -> None:

    """Cette fonction permet d'obtenir un score de f1 tolérante pour évaluer le modèle UGARIT.
    La fonction peut être lancée pour évalué la reconnaissance des lieux ou bien des personnes.
    Il suffit de donner la colonne originelle et une liste avec toutes les valeurs des colonnes UGARIT en arguments."""

    TP = 0
    FP = 0
    FN = 0

    all_actual_values = [value for sublist in actual_values for value in sublist] 
    all_actual_values = [value.replace("[", "").replace("]", "").replace("(", "").replace(")", "") for value in all_actual_values]
    all_predicted_values = [value for sublist in predicted_values for value in sublist]
    all_ner_predicted_values = [value for sublist in ner_predicted_values for value in sublist]

    remaining_actual_values = all_actual_values.copy()
    remaining_predicted_values = all_predicted_values.copy()
    remaining_ner_predicted_values = all_ner_predicted_values.copy()

    for ner_predicted_value in all_ner_predicted_values:
        for actual_value in all_actual_values:
            normalised_predicted_value = ud.normalize("NFD", ner_predicted_value.lower())
            normalised_predicted_value = ''.join(c for c in normalised_predicted_value if ud.category(c) != 'Mn')
            normalised_actual_value = ud.normalize("NFD", actual_value.lower())
            normalised_actual_value = ''.join(c for c in normalised_actual_value if ud.category(c) != 'Mn')
            if normalised_predicted_value == normalised_actual_value:
                TP += 1
                if ner_predicted_value in remaining_ner_predicted_values:
                    remaining_ner_predicted_values.remove(ner_predicted_value)
                if ner_predicted_value in remaining_predicted_values:
                    remaining_predicted_values.remove(ner_predicted_value)
                if actual_value in remaining_actual_values:
                    remaining_actual_values.remove(actual_value)

    if len(remaining_actual_values) > 0:
        FN += len(remaining_actual_values)

    if len(remaining_predicted_values) > 0:
        FP += len(remaining_predicted_values)

  
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    f1_score = (2 * precision * recall) / (precision + recall)
    print(f"Le f1 score tolérant est de {f1_score}.")

                
def get_sound_change_df(df: pd.DataFrame) -> pd.DataFrame:

    """Cette fonction permet d'obtenir une nouvelle dataframe contenant l'écriture
    classique et celle retrouvée dans le texte pour chaque papyrus !"""

    dict_irr = {"old" : [], "new" : []}

    for idx, row in df.iterrows():
        list = row["Text Irregularities"]
        if list == "[]":
            continue

        list = list.strip("[").strip("]").replace("'", "").split(",")
        for pair in list:
            pair = pair.split(":")
            if len(pair) == 2: # je vérifie car certains rows on un truc du genre , read : ou rien read :... je retiens pas ceux là
                old = pair[1].replace("read", "").strip()
                new = pair[0].strip()
                if old != "" and new != "":
                    dict_irr["old"].append(old)
                    dict_irr["new"].append(new)

    sound_change_df = pd.DataFrame(dict_irr)
    return sound_change_df

def normalize_irregularities(cell: str) -> str:

    """Cette fonction permet de normaliser chaque cellule de la dataframe sound_change
    en retirant tous les diacritiques des caractères diacrités."""

    normalised_cell = ud.normalize("NFD", cell)
    normalised_cell = ''.join(c for c in normalised_cell if ud.category(c) != 'Mn')
    normalised_cell = normalised_cell.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
    normalised_cell = normalised_cell.replace("(", "").replace(")", "").replace("<", "").replace(">", "")
    
    return normalised_cell


def get_differences(sound_change_df: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], int], Dict[str, Counter]]:

    """Cette fonction permet d'obtenir différents dictionnaires permettant de représenter
    les différences de graphie du corpus."""

    old_list = sound_change_df["old"].to_list()
    new_list = sound_change_df["new"].to_list()

    dict_diff = defaultdict(list)

    ### J'ai utilisé ndiff, il se pourrait qu'il y avait vraiment plus rapide mais j'ai voulu faire ça manuellement !
    # Les résultats sont très similaires aux vôtres donc j'y crois !

    # ndifff nous permet d'avoir des listes qui comparent deux mots.
    # Nous pouvons avoir ['α', 'π', 'ο', 'λ', '- ε', '- ι', '+ η', '+ μ', 'φ', 'θ', 'ε', 'ι', 'η']
    # Cet exemple montre qu'un ε et un ι sont absents de la nouvelle écriture et qu'ils ont été remplacés
    # par η et μ respectivement. Parfois, un caractère n'est pas remplacé, et parfois un caratère est rajouté
    # sans rien remplacer (pas traité ici.). Au vue de l'exemple, il faut tracer les changements pour trouver
    # quel caractère remplace quoi. Ici, η ne remplace pas ι mais bien ε alors que dans la plupart des cas, car
    # nous n'avons qu'un changement, l'index suivant le caractère remplacé est celui du caractère de remplacement.
    # Il a fallu traiter tous ces cas spécifiques !

    for old, new in zip(old_list, new_list):
        diff = [s.strip() for s in ndiff(old, new)]
        jumps = 0
        for index, char in enumerate(diff):
            # Les caractères - à la suite des uns et des autres
            # sont traités plus bas. Pour éviter les doublons,
            # on saute certaines itérations de la boucle.
            if jumps > 0:
                jumps -= 1
                continue

            if "-" in char:
                if index == len(diff) - 1 :
                    # Si dernier caractère alors remplacé par rien
                    dict_diff[char.strip("-").strip()].append("")
                else:
                    if "-" in diff[index + 1]: # Séquence de caractères - PAIRE απολειφθειη απολημφθειη
                        j = 0
                        num_replacements = 1
                        while index + j < len(diff) - 1 and "-" in diff[index + j]: # differences :  ['α', 'π', 'ο', 'λ', '- ε', '- ι', '+ η', '+ μ', 'φ', 'θ', 'ε', 'ι', 'η']
                            num_replacements += 1
                            if index + num_replacements < len(diff):
                                if "+" in diff[index + num_replacements]:
                                    dict_diff[diff[index + j].strip("-").strip()].append(diff[index + num_replacements].strip("+").strip())
                                    j += 1
                                    jumps += 1
                            else:
                                dict_diff[diff[index].strip("-").strip()].append("")
                                break
              
                    i = 1
                    # Pour les caractères remplacés par rien au sein du mot.
                    if "+" not in diff[index + i] and "-" not in diff[index + i]:
                        dict_diff[char.strip("-").strip()].append("")
                        break
                    # Pour les caractères remplacés !
                    elif "+" in diff[index + i]:
                        replacement_str = ""
                        while index + i <= len(diff) - 1 and "+" in diff[index + i]:
                            replacement_str += diff[index + i].strip("+").strip()
                            i += 1
                        dict_diff[char.strip("-").strip()].append(replacement_str)

    ### Je fais ça seulement pour faciliter les questions d'après ! 
    dict_differences = {} # le premier dico a comme clé une paire de caractère et son remplacement et en valeur le nombre d'occurrences de ce changement.
    dict_differences_by_grapheme = {}

    for char, replacements in dict_diff.items():
        replacements_count = Counter(replacements)
        dict_differences_by_grapheme[char] = replacements_count # le deuxième a un grapheme en clé et en valeur un dico Counter 
        # avec tous les caractères qui le remplacent et keurs occurrences de remplacement

        for replacement, count in replacements_count.items():
            dict_differences[(char, replacement)] = count

    
    return dict_differences, dict_differences_by_grapheme

def get_ten_most_common_changes(dict_differences: Dict[Tuple[str, str], int]) -> None:

    """Cette fonction permet d'obtenir les dix changments de graphie les plus communs"""

    changes = sorted(dict_differences.items(), key=lambda x: x[1], reverse=True)
    for change in changes[:10]:
        print(change)

def get_thirty_changes(dict_differences_by_grapheme: Dict[str, Counter]) -> Dict[str, int]:

    """Cette fonction permet d'obtenir les caractères remplacés plus de 30 fois dans le corpus"""

    thirty_changes = {}

    for grapheme, replacements in dict_differences_by_grapheme.items():
        thirty_changes[grapheme] = sum([count for key, count in replacements.items() if key != ""]) ### J'ai fait en sorte de ne pas compter les moments ou on a un cractère remplacé par rien
        # Mais du coup... j'en ai 9 et pas 8 oups

    thirty_changes = {grapheme: count for grapheme, count in thirty_changes.items() if count > 30}
    for grapheme, count in thirty_changes.items():
        print(grapheme, count)

    return thirty_changes
  
def lets_make_some_pie(thirty_changes: Dict[str, int], dict_differences_by_grapheme: Dict[str, Counter]) -> None:

    """Cette fonction permet de visualiser les changements opérés sur 
    les graphèmes remplacés plus de trente fois."""

    num_charts = len(thirty_changes)
    cols = 3
    rows = (num_charts // cols) + 1
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    axs = axs.flatten()

    for grapheme, replacements in dict_differences_by_grapheme.items():
        if grapheme in thirty_changes.keys():
            labels = list([key for key in replacements.keys() if key != ""])
            sizes = list([values for key, values in replacements.items() if key != ""])
            
            ax = axs[list(thirty_changes.keys()).index(grapheme)]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title(f"Old sound: {grapheme}")

    for ax in axs[num_charts:]:
      ax.axis('off')

    plt.tight_layout()
    plt.show()

def graph_corrections(dict_differences_by_grapheme: Dict[str, Counter]) -> Network:

    """Cette fonction permet de créer un graph représentant les relations de remplacement."""

    # on enlève les noeuds ou un caractère est remplacé par rien pour le graph.
    # sinon c'est un peu perturbant
    dict_differences_by_grapheme = {grapheme : {replacement : count for replacement, count in replacements.items() if replacement != ""} for grapheme, replacements in dict_differences_by_grapheme.items()}
    dict_differences_by_grapheme = {grapheme : replacements for grapheme, replacements in dict_differences_by_grapheme.items() if replacements != {}}

    graph = Network(height="800px", width="800", directed=True, bgcolor="#222222", font_color="white")

    for grapheme, replacements in dict_differences_by_grapheme.items():
        graph.add_node(grapheme, size=20, label=grapheme) 
        for replacement, weight in replacements.items():
            graph.add_node(replacement, size=10, label=f"'{replacement}'", color="pink")
            graph.add_edge(grapheme, replacement, value=weight, title=f'corrections : {weight}')

    return graph



def main():

    parser = argparse.ArgumentParser(
    description="""Ce script correspond aux questions de la partie II sur l'analyse du dataset (parties 4 à 8)
                J'ai décidé d'inclure une gestion des arguments pour que vous puissiez accéder aux outputs des 5
                parties plus clairement. Bonne lecture et bonne correction ! καλὴ ἐλπίς (traduction surement honteuse)"""
    )
    parser.add_argument("-P", "--partie", required=True, choices=["4", "5", "6", "7", "8", "streamlit"],
                        help="""Vous pouvez choisir la partie dont vous désirez l'ouput ! \n
                        Exemple d'utilisation : python3 corpus_analysis.py -P 4.""")
    
    args = parser.parse_args()

    # les fonctions essentielles pour toutes les parties
    df_raw = read_data("../data/tables/papyrus_corpus.csv")
    df_cleaned, how_many_null_texts, after_len = clean_df(df_raw)
    df = sort_df(df_cleaned)

    if args.partie == "4":
        print("RÉPONSES POUR LA PARTIE 4 : \nNETTOYAGE DU DATASET\n\n")
        print("1 - Charger le fichier csv dans un DataFrame :\n")
        print(df_raw.head(), "\n\n")
        print("2 - Observer le dataset. Que dire des 4 premières lignes ? Que faire ? Faites-le.\n")
        print("\tLes quatre premiers papyri n'ont pas de texte. Nous allons donc les retirer.\n\n")
        print("3 - Combien de textes n'ont pas été capturés pendant le scraping ? Comment le voit-on ? Enlevez-les.\n")
        print("\tNous pouvons retirer les papyri sans texte à l'aide des filtrages de Pandas,\n\tnous filtrons les lignes avec un texte null et dropons ces dernières.")
        print(f"\tNous avons retiré {how_many_null_texts} papyri.\n\n")
        print("4 - Combien la collection compte-t elle de papyrus après nettoyage ?\n")
        print(f"\tNous nous retrouvons avec {after_len} papyri dans notre collection.\n\n")
        print("5 - Trier la collection selon l'ID (ordre croissant).\n")
        print("\t",print(df.head(10)))

    date_intervals = formating_dates(df)
    get_provenance(df)
    if args.partie == "5":
        print("RÉPONSES POUR LA PARTIE 5 : \nETUDE DE CORPUS : GENRE, LIEU ET DATE\n\n")
        print("1 - Affichez les différents genres sur une piechart :\n")
        get_content(df)
        print("2 - Combien de papyri ont-ils été réutilisés ?\n\n")
        num_recycled_papyri = recycled_papyri(df)
        print(f"\t{num_recycled_papyri} ont été ré-utilisés !\n\n")
        print("3 - D'où viennent les papyri ? De même ne retenez que le nom de la ville. Faites un diagramme en barre cette fois.\n\n")
        visualise_provenance(df)
        print("4 - Qu'en concluez-vous ?\n")
        print("\tNous pouvons observer que les papyri proviennent en écrasante majorité d'Aphrodito.\n\n")
        print("5 - Formattez les dates sous le format d'une date simple AD xxx ou d'un intervalle AD xxx - xxx.\n")
        print("\tJ'obtiens des intervalles de forme :")
        for interval in date_intervals[:10]:
            print("\t", interval)
        print("\n")
        print("6 - Utilisez ensuite ces valeurs discrètes ou intervalles pour construire un diagramme représentant la densité de papyri sur chaque année du dataset.\n")
        density_maps_of_dates(date_intervals)
        print("\tTout comme les villes de provenance, les papyri datent en majorité du siècle 500-600.")

    # Fonctions obligatoires pour continuer les parties suivantes
    clean_text(df)

    df["People List Clean"] =  df["People List"].apply(column_clean_function_people)
    df["Places List Clean"] = df["Places List"].apply(column_clean_function_places)

    if args.partie == "6":
        print("RÉPONSES POUR LA PARTIE 6 : \nNETTOYAGE DU TEXTE GREC\n\n")
        print("1 - écrivez une première fonction de nettoyage du texte qui retire les chiffres arabes,\nles lignes perdues | gap | ainsi que les caractères spéciaux '†' et '⳨'.\nRetirez les crochets et paranthèses puis stockez les portions incertaines dans une colonne.\n")
        print("\tLe texte nettoyé :")
        print("\t", df["Full Clean"], "\n")
        print("\tColonne incertaine :")
        print("\t", (df["Uncertain Portion"]), "\n\n")
        print("2 - Combien y a-t de papyrus dont plus du tiers du texte est incertain ?")
        uncertain_texts = how_many_too_uncertain_texts(df)
        print(f"\tIl y a {uncertain_texts} textes dont le tiers du contenu est incertain !")

    elif args.partie == "7":
        get_entities(df)
        print("RÉPONSES DE LA PARTIE 7 : \nIDENTIFIER LES NOMS DE PERSONNES ET DE LIEUX\n\n")
        print("1 - Observez le contenu des cellules de la case 'people-list'.\nQue remarquez-vous ? Réglez le(s) problème(s) de manière à ne retenir que les noms.\n")
        print("\tLe contenu de la colonne est pollué par des chiffres arabes et de l'anglais.\n\tCes éléments faisaient partie des enfants de la balise contenant les\n\tnoms mentionnés sur le site.\n\tPour les retirer, nous ne retenons que les caractères grecs\n\tet enlevons les chiffres arabes.")
        print("\tLa colonne nettoyée :")
        print("\t", df["People List Clean"])
        print("\tOn peut aussi faire de même pour la colonne 'Places List' qui contient des dictionnaires.")
        print("\tLa colonne nettoyée :")
        print("\t", df["Places List Clean"], "\n\n")
        print("2 - Téléchargez le modèle de NER suivant sur Hugging Face https://huggingface.co/UGARIT/grc-ner-bert\nUtilisez le pour stocker les entités repérées dans les colonnes 'People Ugarit'\n'Places Ugarit' et 'Other Ugarit' Commentez les résultat.\n")
        print("\t", df["People Ugarit"])
        print("\t", df["Places Ugarit"])
        print("\t", df["Other Ugarit"])
        print("\t Les NE reconnues par UGARIT sont parfois perturbées par des signes de ponctuations\net sont généralement moins nombreuses que pour celles que nous avons originellement.\nPour améliorer les prédictions, je vais créer une nouvelle fonction de nettoyage du texte\ndans laquelle j'enlève les points, tirets et slashs.\nJe normalise aussi pour éviter les soucis d'accents.\n(En réalité j'ai appliquée cette fonction plus tôt pour éviter de relancer le modèle deux fois!)")
        print("3 - Calculez le F1 score du système de NER sur notre corpus : \nde façon sévère (la catégorie de l'entité importe) \net de façon tolérante (la catégorie de l'entité n'importe pas)\n")
        get_f1_score_severe(df["Places List Clean"].to_list(), df["Places Ugarit"].to_list())
        get_f1_score_tolerant(df["Places List Clean"].to_list(), df["Places Ugarit"].to_list(), df["Places Ugarit"].to_list() + df["Places Ugarit"].to_list() + df["Other Ugarit"].to_list())
        print(f"Les F1 scores sont assez elevés après nettoyage et normalisation !\nJe dois vous avouer que je trouve ça un peu suspect mais avant la normalisation j'avais des scores de\n0.06 et 12 donc c'est peut-être juste la magie des diacritiques.")

    sound_change_df = get_sound_change_df(df)

    if args.partie == "8":
        print("RÉPONSES DE LA PARTIE 8 : \nETUDE DES FAUTES DE GRAPHIE\n\n")
        print("1 - Créez un nouveau DataFrame nommé 'sound_change_df' qui aura pour colonne\n'old' (forme correcte en grec classique) et 'new' (forme trouvée dans le papyrus)\n\n”")
        print("\t", sound_change_df, "\n\n")
        print("2 - Normalisez les irrégularités afin de retirer les diactritiques.\n\n")
        sound_change_df["old"] = sound_change_df["old"].apply(normalize_irregularities)
        sound_change_df["new"] = sound_change_df["new"].apply(normalize_irregularities)
        print("\t", sound_change_df, "\n\n")
        print("3 - Quels sont les 10 changements les plus fréquents ?\n")
        dict_differences, dict_differences_by_grapheme = get_differences(sound_change_df)
        print("\tLes dix changements les plus fréquents sont les suivants :")
        get_ten_most_common_changes(dict_differences)
        print("\n")
        print("4 - Quels graphèmes du grecs classiques ont été modifiés plus de 30 fois dans le dataset ?\n")
        print("\tLes graphèmes modifiés plus de 30 fois sont :")
        thirty_changes = get_thirty_changes(dict_differences_by_grapheme)
        print("\n")
        print("5 - Créez un graphique unique qui représente pour chacun de ces 8 graphèmes\nla nouvelle forme qu'il va prendre sous la forme d'un pie chart.\n\n")
        lets_make_some_pie(thirty_changes, dict_differences_by_grapheme)
        print("BONUS - Représentez le graphe de conversion de sons.\n")
        graph = graph_corrections(dict_differences_by_grapheme)
        graph.show_buttons(filter_=True)
        graph.show("../data/graph/corrections.html")

    df.to_csv("../data/tables/clean_papyrus-corpus.csv")
    if args.partie == "streamlit":
        df = pd.read_csv("../data/tables/clean_papyrus-corpus.csv")
        df = df.drop(["Authors / works", "Book form", "Content (beta!)", "Culture & genre", "Language/script", "Material", "Note", "People List", "Places List", "Recto/Verso", "Reuse note", "Reuse type", "Ro", "Uncertain Portion"], axis=1)
        df.to_csv("../data/tables/clean_papyrus-corpus.csv")

if __name__ == "__main__":  
    main()