from bs4 import BeautifulSoup
import requests
import pandas as pd
from typing import List, Dict

"""Ce programme permet de scrapper la collection de papyri du site
    `trismegistos` afin d'obtenir un fichier csv (papyri.csv) les 
    répértoriant tous (mon précieux...) !
    Lancez-moi pour tester le scraping sur les 10 premiers papyri !"""

def get_papyri_ids(csv_file: str) -> List[str]:
    """Fonction pour obtenir la liste des id des papyrus à partir du CSV !"""

    df = pd.read_csv(csv_file)
    tm_id_list = df["ID"].to_list()

    return tm_id_list


def get_page_data(page_url: str) -> BeautifulSoup:
    """Fonction pour récupérer les données de chaque page."""

    page = requests.get(page_url)
    page_data = BeautifulSoup(page.content, 'lxml')

    return page_data 


def get_papyrus_page(home_url: str, home_data:BeautifulSoup, tm_id: str) -> str:
    """Fonction pour obtenir l'url de chaque page !"""

    ### On met l'ID dans la barre de recherche et on clique ! 
    form_tag = home_data.find("form")
    target_url = home_url + form_tag["action"]
    for child in form_tag.children:
        if child.name == "input":
            key = child["name"]
    
    pair = {key : tm_id}

    page = requests.post(target_url, pair)
    page_url = page.url
    
    return page_url

def get_url_list(tm_id_list: List[str], home_url: str, home_data: BeautifulSoup) -> List[str]:
    """Fonction pour obtenir une liste d'urls ou chaque
    url correspond à une page individuelle de papyrus."""

    url_list = []
    for tm_id in tm_id_list:
        page_url = get_papyrus_page(home_url, home_data, tm_id)
        url_list.append(page_url)
    
    return url_list


def scrap_papyrus(page_data: BeautifulSoup, page_url: str) -> Dict:
    """Fonction pour obtenir l'ensemble des données désirées pour chaque papyrus."""

    ### On initialise le dictionnaire qui contient toutes les infos de notre papyrus.
    papyrus_info = {"ID" : "", "Date" : "", "Provenance" : "", "Language/script" : "", "Material" : "",
                    "Content" : "", "Publications" : [], "Collections" : [], "Archives" : [], "Text" : "",
                    "People" : [], "Places" : [], "Text irregularities" : [], "Geo" : ""}

    papyrus_info["ID"] = page_url.split("/")[-1]

    ### Pour récupérer Date, Provenance, Language, Material et Content
    # On fait des split join strip de la décadence pour vraiment avoir les
    # données qui nous intéressent ! 
    papy_details_div = page_data.find("div", id="text-details", class_="text-info")
    for child in papy_details_div.children:
        if "Date" in child.text:
            date = (child.text).split(":")[1].strip() 
            papyrus_info["Date"] = date                 
        elif "Provenance" in child.text:               
            provenance = (child.text).split(":")[1].strip()
            papyrus_info["Provenance"] = provenance
        elif "Language" in child.text:
            language = child.text.split(":")[1].strip()
            papyrus_info["Language/script"] = language
        elif "Material" in child.text:
            material = child.text.split(":")[1].strip()
            papyrus_info["Material"] = material
        elif "Content" in child.text:
            content = [x for index, x in enumerate((child.text.split(":"))) if index != 0]
            content = ":".join(content)
            papyrus_info["Content"] = content
        
    ### Pour récupérer Publications
    papy_publs_div = page_data.find("div", id="text-publs", class_="text-info")
    for child in papy_publs_div.children:
        if child.text != "Publications" and not child.text.isspace():
            publication = (child.text).split("Please")[0].strip()
            papyrus_info["Publications"].append(publication)

    ### Pour récupérer Collections
    papy_coll_div = page_data.find("div", id="text-coll", class_="text-info")
    for child in papy_coll_div.children:
        if child.text != "Collections" and not child.text.isspace():
            collection = (child.text).strip()
            papyrus_info["Collections"].append(collection)

    ### Pour récupérer les archives lo
    papy_arch_div = page_data.find("div", id="text-arch", class_="text-info")
    for child in papy_arch_div.children:
        if child.text != "Archive" and not child.text.isspace():
            archive = (child.text).strip()
            papyrus_info["Archives"].append(archive)

    
    ### Pour obtenir le texte 
    text = ""
    papy_words_div = page_data.find("div", id="words", class_="text")

    # On retire les span pour n'avoir que les mots
    for span in papy_words_div.find_all("span"):
        span.extract()

    # On concatène tous les mots sans prendre en compte les chiffres
    # ou les indications 'omit', 'add'...
    for a in papy_words_div.find_all("a", class_="info-tooltip"):
        if not a.text.isspace() and not a.text.isdigit():
            text += " " + a.text

    # On retire les symboles que no queremos
    text_list = text.split("†")
    text_list = [x for index, x in enumerate(text_list) if index != 0]
    text = (" ".join(text_list)).strip()
    papyrus_info["Text"] = text

    ### Pour récupérer les people mentionnés
    people = []
    papy_people_div = page_data.find("div", id="people", class_="text")
    for child in papy_people_div.find_all("li", class_='item-large'):
        people.append(child.text)

    papyrus_info["People"] = people

    ### Pour récupérer les places mentionnées
    # On en profite pour récupérer le num geo-ref
    # qui nous permet ensuite d'accéder aux pages
    # de chaque lieu pour télécharger les fichiers json
    places = []
    geo = []
    papy_places_div = page_data.find("div", id="places", class_="text")
    for child in papy_places_div.find_all("li", class_='item-large'):
        if child:
            places.append(child.text)
            geo_adress = child.get("onclick")
            geo_adress = geo_adress.strip(")").strip("getgeo").strip("(")
            page_base_url = page_url.split("text")
            geo_url = f"{page_base_url[0]}/georef/{geo_adress}"
            if geo_url not in geo:
                geo.append(geo_url)
    
    papyrus_info["Geo"] = geo
    papyrus_info["Places"] = places
    
    ### Pour récupérer les irrégularités ! 
    irregularities = []
    papy_irregularities_div = page_data.find("div", id="texirr", class_="text")
    for child in papy_irregularities_div.find_all("li", class_='item-large'):
        irregularities.append(child.text)

    papyrus_info["Text irregularities"] = irregularities

    return papyrus_info

### Après pas mal d'essais et de recherche (et j'avoue que j'ai aussi demandé à ChatGPT pour ça),
# j'ai cru comprendre que les éléments révélés sur une page suite à un event JS ne peuvent pas être
# obtenus grâce à Beautiful Soup car la librairie ne permet que de scraper l'HTML statique !
# Je ne trouvais pas les numéros TM Geo attribués à une localisation car ces derniers étaient hidden.
# ChatGPT m'a expliqué le soucis des events JS et effectivement, quand je regarde la page source,
# on a bien un event JavaScript ! Quelques semaines après ce script, j'ai pu utiliser Sélénium pour la première fois
# et la librairie aurait été bien utile (j'avais des soucis de driver donc impossible de l'utiliser avant...).
# J'ai plutôt récupéré les Geo-Ref numbers qui nous amènent sur les pages de mention du lieu. Sur ces pages, 
# je peux obtenir les numéros TM ! Donc on prend un chemin un peu plus long que si on déclenchait l'event. 

def get_json_file(id: str, geo_page: BeautifulSoup, geo_url: str) -> None:
    """Fonction pour obtenir les fichiers json de chaque lieu mentionné
    dans le texte d'un papyrus."""

    ### On ouvre la page Geo-Ref et on récupére les liens des pages 
    ### de chaque lieu.
    ### On a en bas de chaque page une liste d'autres lieux mentionnés
    ### dans le texte concerné. Je fais l'opération plusieurs fois
    ### mais on retire les doublons avec une condition et on est au 
    ### moins sûres de ne pas avoir oublié de lieux !
    geo_links = []
    table = geo_page.find("tbody")
    links = table.find_all('a', class_='unicode')
    for child in links:
        href_value = child.get("href")
        geo_link = geo_url.split("georef")[0].strip("/") + href_value.strip("..")
        if geo_link not in geo_links:
            geo_links.append(geo_link)

    for link in geo_links:
        link_page = requests.get(link)
        link_page = BeautifulSoup(link_page.content, "lxml")
        json_file_div = link_page.find("div", class_="infobox infobox-JSON flex-items")
        for child in json_file_div.find_all("a"):
            if child.text == "Download":
                place_id = link.split("place")[-1].strip("/")
                download_link = link.split("place")[0] + child.get("href")
                file_response = requests.get(download_link)
                file_name = f"{id}_{place_id}"
                with open(f"../data/json_files/{file_name}.json", "wb") as file:
                    file.write(file_response.content)


def main():

    tm_id_list = get_papyri_ids("../data/tables/papyrus_metadata.csv")
    home_url = 'https://www.trismegistos.org/'
    home_data = get_page_data(home_url)
    print("On récupère les urls...")
    url_list = get_url_list(tm_id_list[:10], home_url, home_data)
  
    all_papyri_info = {"ID" : [], "Date" : [], "Provenance" : [], "Language/script" : [], "Material" : [],
                    "Content" : [], "Publications" : [], "Collections" : [], "Archives" : [], "Text" : [],
                    "People" : [], "Places" : [], "Text irregularities" : [], "Geo" : []}

    for page_url in url_list[:10]:
        print(f"Ça scrap fort la page {page_url.split('/')[-1]}...")
        page_data = get_page_data(page_url)
        papyrus_info = scrap_papyrus(page_data, page_url)
        for cat, info in papyrus_info.items():  
            print(f"{cat}:{info}\n")
        print("#####################################################")      
        for key in all_papyri_info.keys():
            all_papyri_info[key].append(papyrus_info[key])
            
        if papyrus_info["Geo"] != []:
            for geo_url in papyrus_info["Geo"]:
                geo_page = get_page_data(geo_url)
                get_json_file(papyrus_info["ID"], geo_page, geo_url)

    total_df = pd.DataFrame(all_papyri_info)
    total_df.to_csv("../data/tables/papyri.csv")

if __name__ == "__main__":
    main()