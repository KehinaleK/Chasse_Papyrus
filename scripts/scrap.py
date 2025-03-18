import pandas as pd
import bs4
from bs4 import BeautifulSoup #Utiliser pour l'extraction des contenus textuels
import requests

def map_dataframe():

    dict_map = {"Lieu" : [], "Lat" : [], "Lon" : []}
    df = pd.read_csv("../data/tables/papyrus_corpus.csv")
    places = df["Places List"].to_list()
    
    i = 0
    for set in places:
        if set != "{}":
            set = set.strip("{").strip("}").split(",")
            for pair in set:
                pair = pair.split(":")
                if len(pair) == 2:
                    name = pair[0].strip().strip("'").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("-", "").replace(".", "").replace("/", "").replace("{", "").replace("}", "")
                    id_geo = pair[1].strip().strip("'")
                    geo_url = f"https://www.trismegistos.org/georef/{id_geo}"
                    geo_page = requests.get(geo_url)
                    if geo_page.status_code == 200:
                        geo_page = BeautifulSoup(geo_page.content, "lxml")
                        coord_div = geo_page.find("div", id ="right-infobox")
                        for child in coord_div.children:
                            if "Lat,Long" in child.text:
                                coor = (child.text).strip("(Lat,Long)").strip()
                                lat = coor.split(",")[0]
                                lon = coor.split(",")[1]
                                dict_map["Lieu"].append(name)
                                dict_map["Lat"].append(lat)
                                dict_map["Lon"].append(lon)
                                print(name, lat, lon)
        print(i)
            
        i += 1

    df_coord = pd.DataFrame(dict_map)
    df_coord.to_csv("../data/tables/coordonnees.csv")

map_dataframe()