# [Imports]

from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import requests
import tempfile
import spacy
from datetime import date, datetime
import locale
# locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
from threadpoolctl import threadpool_limits
import streamlit as stm

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# [Fonctions]


from_text_extract_number= lambda x : int(re.search(r"\d+", x).group(0)) # Mini-fonction d'extraction des nombres contenu dans une chaine de caractère
from_ocr_to_text= lambda x : re.sub(r'\s+', ' ', pytesseract.image_to_string(x, lang='fra')).lower() #Mini-fonction de transformation les documents image en texte en les océrisant


def select_url(start_date, end_date):
    """Fonction de récupération des liens de téléchargement de tout les recueils. 
    Il prend en entrée l'intervalle de date de publication des RAA et retourne une dataframe constitué de trois colonnes dont l'url , 
    la date de publication et la référance du RAA """
    
    principale_url= "https://www.prefectures-regions.gouv.fr" 
    url_page= f"{principale_url}/bretagne/Documents-publications/Recueils-des-actes-administratifs/Recueil-des-actes-administratifs"  # Lien url vers la page ou se trouve les RAA de la préfecture de la régioin de bretagne 

    # #user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
    # user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'  # Paramètre de requête 

    href_and_date= pd.DataFrame()

    # # Requête de récupération du contenu des balises spécifiques de la page(scrapping)
    # my_req = Request(url_page,headers={'User-Agent': user_agent}) 
    # response = urlopen(my_req, timeout=20)
    # res_final= response.read()

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url_page)
    html = driver.page_source

    bsObj = BeautifulSoup(html, "lxml")
    driver.quit()

    # bsObj = BeautifulSoup(res_final, "lxml") # en utilisant le parser de lxml
    link_download_series= pd.Series(bsObj.find_all("a", class_= "link-download"))
    href_and_text= link_download_series.map(lambda x: (x.attrs["href"], x.text)) # récuperation des liens et du texte de la balise "a" de classe "link-download" sous forme de tuple
    href_and_text= link_download_series.map(lambda x: (x.attrs["href"], re.search(r"du\s+(\d{1,2}\w*)\s+([a-zéàèû]+)(?:\s+(\d{4}))?|(\d{1,2})\/(\d{1,2})\/(\d{4})", x.text.lower()))) # application d'expression régulière pour la récupération de la référence du recueil
    
    # Récupération , netoyage de la référence des RAA et formatage des valeurs manquantes
    href_and_text= href_and_text.map(lambda x: (x[0], x[1].group().strip()) if x[1] is not None else (None, None)) 
    href_and_text= href_and_text.map(lambda x: (x[0], x[1] + ' ' +  re.search("(?<=-)\d{4}(?=-)",x[0]).group()) if x[1] is not None and re.search("\d{4}", x[1]) is None else (x[0], x[1]))
    
    # Mise en base des donnée (dimension dataframe pandas)
    href_and_date["url"], href_and_date["date"]= principale_url + href_and_text.map(lambda t: t[0] ), href_and_text.map(lambda t: t[1] )
    href_and_date= href_and_date[href_and_date["date"].map(lambda h : h is not None)]

    # Formatage de la date de publication  en objet datetime
    href_and_date["date"]= href_and_date["date"].map(lambda x: x.replace("du ", ''))
    href_and_date["date"]= href_and_date["date"].map(lambda x: x.replace('1er', '1'))
    href_and_date["date"]= href_and_date["date"].map(lambda x: x.replace('fe', 'fé'))
    href_and_date["date"]= href_and_date["date"].map(lambda x: x.replace('de', 'dé'))
    href_and_date["date"]= href_and_date["date"].map(lambda x: datetime.strptime(x, "%d/%m/%Y").date() if re.search("\s\d{4}", x) is None  else datetime.strptime(x, "%d %B %Y").date() )
    href_and_date= href_and_date.sort_values('date', ascending= False) # Trie du dataframe par date de publication
    href_and_date= href_and_date[href_and_date["date"].map(lambda y :  y<= end_date and y>= start_date )]

    # Formatage du nom du recueil
    href_and_date["ref_recueil"]= href_and_date["url"].map(lambda u :  re.search(r"recueil-(r\d+-\d+-\d+)", u).group(1)) 
    
    return href_and_date


def extract_sommaire(pages):
    """Fonction d'extraction du sommaire d'un recueil d'acte administratif, 
    il prend en argument les pages de l'arrêté et retourne le sommaire du document extrait"""
    
    sommaire_first_page= re.sub(r'\s+', ' ', pytesseract.image_to_string(pages[1], lang='fra')).lower() # Recuperation de la première page du sommaire 
    start_first_arr= from_text_extract_number(re.findall(r"page.\d+", sommaire_first_page)[0]) # Retrouve le numero de la page de début du premier arrété du recueil
    sommaire_page_end=  start_first_arr - 1   # Calcule le nombre associé  à la page de fin du sommaire
    sommaire= '' # initialise un sommaire vide 

    # Récupération du le sommaire du RAA par coupure 
    for page_num, page in enumerate(pages, start=1) :  # la boucle de lecture s'arrçete à la page de fin du sommaire 
        sommaire+= from_ocr_to_text(page)  
        if page_num== sommaire_page_end:
            break 
    sommaire= re.search('(?<= somm).*?(?=$)',sommaire).group()

    return sommaire


def extract_component_arr(pages, tuple_start_tot_page_arr, component):
    """Fonction d'extraction des article d'un recueil d'actes administratif, 
    qui prend en entré une page et un tuple de la forme (numero de début de page de l'article,
    numéro de fin de page de l'arrêté et la nature du composent à extraire «"articles" ou "visas"») et 
    retourne soit la liste des numéro d'article et la liste des articles si component= articles ou 
    retourne le bloc de texte relative au visas"""

    start_arr, total_page_arr= tuple_start_tot_page_arr # Récupération du numéro de début et du total de page du de l'arrêté
    pages_arr= ''

    # Récupération des chaines de caractère sur toutes les pages de l'arrêté
    for i in range(start_arr - 1, start_arr + total_page_arr): 
        pages_arr+= from_ocr_to_text(pages[i])
    
    if component== "articles":
        article_texte = re.search(r"(arrete\s:|arretent\s:|arrete.article|arretent.article|décide\s:|decide\s|decide.article|décide.article).*?(fait à.*|$)", pages_arr) # récupère tout les articles dans l'arreté étudié en un bloc concise de texte
        liste_art = re.findall(r"(?<!\S)article\s\d+\w* ?:?.*?(?=(?<!\S)article\s\d+\w* ?:?|fait à|$)", article_texte.group(0)) if article_texte is not None else [] # Récupère chaque article de l'arreté  en liste d'article
        
        if liste_art!= []:
            liste_num_art= [int(re.findall(r"article\s(\d+)\w*\s?:?", article)[0]) for article in liste_art] # Recupère le numéro de l'article 
            liste_art= [re.findall(r"(?<=(?<!\S))(.*)", article)[0].strip() for article in liste_art] # Rassemble tout les articles en liste
        else:
            liste_num_art, liste_art= np.nan, np.nan #Attribut la valeur null si l'article est inexistant
        
        return liste_num_art, liste_art 
    
    elif component== "visas":
        all_vu= re.findall("vu\s.*?(?=arrete\s:|arretent\s:|arrete.article|arretent.article|decide\s|decident\s|décide\s:|décide.article|decide.article|\sfait\sà|$)", pages_arr) #récupère tout les visas
        if all_vu== []:
            all_vu= np.nan # Attribution de la valeur null si inexistant de visas

        else:  # Fait des traitements de texte(mise en forme) sur les visas récupérés
            all_vu= all_vu[0].replace(" ; vu", ". vu")  
            all_vu = all_vu.replace("; - vu", ". vu") 
            all_vu = all_vu.replace(";__ vu", ". vu") 
            all_vu = all_vu.replace("; considérant", ". considérant") 

        return all_vu 


def extract_infos_signatures(pages, tuple_start_tot_page_arr):
    """Fonction qui recupère les informations témoins du document soit la ville ,
    la date de signature ainsi que les signataire, qui prend en entré une page et 
    un tuple de la forme (numero de début de page de l'article,numéro de fin de page de l'arrêt"""
    
    start_arr, total_page_arr= tuple_start_tot_page_arr
    pages_arr= ''
    for i in range(start_arr - 1, start_arr + total_page_arr):  
        pages_arr+= from_ocr_to_text(pages[i]) # Récupère l'article en entier car ce n'est pas forcément sur la dernière page que'est imposé les cachés et signatures(nous pouvons avoir des sauts de page parfois)
    
    #Récupération de la ville ou remplacement par la valeur nulle si mauvais format non lisible
    ville= re.findall(r"(?<=fait\sà\s).*?(?=,|\s)", pages_arr)
    ville=  np.nan if ville==[] else ville[0]

    # Récupère uniquement la partie en rapport avec la date de signature et fait une extraction de la date
    date_signature= re.findall(r"fait à.*?(\d{1,2}\s+[a-zéàè]+(?:\s+\d{4})?)", pages_arr) 
    date_signature = np.nan if date_signature==[] or bool(re.match(r"^\d{1,2}\s+[a-zéàèûôêîâçù]+\s+\d{4}$",date_signature[0].strip()))==False else date_signature[0].strip()

    # Récupère uniquement la partie en rapport avec l'identité des signataires(nom & prénom) ou remplacement par la valeur nulle si mauvais format non lisible
    signataires_text= re.findall(r'fait\sà\s.*?(?=r\d{2}\-\d{4})', pages_arr)  # récupère la chaine de caractère entre la ville de signature et la date
    if signataires_text==[]:
        signataires= np.nan 
    else :
        # Reconnaissance d'Entités Nommées(NER)
        nlp = spacy.load("fr_core_news_sm") # Charge le modèle français de spaCy
        doc = nlp(signataires_text[0].strip()) # Applique le traitement NLP
        signataires = [ent.text for ent in doc.ents if ent.label_ == "PER" and "www" not in ent.text] if isinstance(signataires_text, str) else np.nan  # Reconnaissance et extraction d'entités nommées de type "PERSON"        
    
    return ville,date_signature, signataires



def pages_to_df(pages):
    """Fonction de mise génération d'une database récapitulatif du receuil.
    Prend en entrée les pages d'un arrêté et extrait des informations 
    spécifiques (le contenu ) de façon catégorisé dans une base de donnée"""

    df_text= pd.DataFrame()

    page_garde= pytesseract.image_to_string(pages[0], lang='fra').replace("\n", " ").lower() # Recupère le texte de la page de garde 
    id_rc= re.search(r"(?<=n°)[a-z0-9-]+", page_garde).group().strip() # Extraction de la référence du recueil
    date_pub= re.search(r"\d{1,2}\s+[a-zéàè]+\s+\d{4}", page_garde).group(0) # Extraction de la date de publication du recueil 
    sommaire= extract_sommaire(pages) # Extraction du sommaire du recueil
    start_pages_rc= re.findall(r"page.\d+", sommaire) # Extraction du numèro de la page de début du recueil
    total_page_rc= re.findall(r"\(\d+\s+pages?", sommaire) # Extraction de la page de fin  du recueil
    titres_arrete = re.findall(r"(r\d[\s\S]*?)(?=\s+\(\d+\s+pages\))", sommaire) # Extraction du titre des arrétés dans le sommaire
    interval_p_rc= [(from_text_extract_number(start_p),from_text_extract_number(total_p)) for start_p, total_p  in zip(start_pages_rc, total_page_rc)] # Extraction de 
    
    liste_objet_arrete= []

    with threadpool_limits(user_api='openmp', limits=-1): # Assignation de tout les coeurs du pc pour la taâche suivante
        
        for num_arr, titre_ar in enumerate(titres_arrete):

            liste_num_art, liste_article_arr= extract_component_arr(pages, interval_p_rc[num_arr],component= "articles") # Récupération des numéros des artcles  et de l'ensemble des articles sous forme de listes
            ville,date_signature, signataires= extract_infos_signatures(pages, interval_p_rc[num_arr]) #Récupération des information géo-temporale et l'identité des signataire de l'arrêté
            total_page_arr= interval_p_rc[num_arr][1]

            ttr_rec= f"est issue du receuil administratif {id_rc} " if not(isinstance(id_rc, float)) else " "
            d_pub_rec= f"publier le {date_pub}. " if not(isinstance(date_pub, float)) else " "
            ttr_arr= f"Cette arrêté est titré {titre_ar} " if not(isinstance(titre_ar, float)) else " "
            T_page_arr= f"Cette arrêté est constitué de {total_page_arr} pages " if not(isinstance(total_page_arr, float)) else ""
            v_sign_arr= f",signé dans la ville {ville} " if not(isinstance(ville, float)) else " "
            signs_arr= f"par des signataires ; {signataires} " if not(isinstance(signataires, float)) else " "
            dt_sign_arr= f" le {date_signature}." if not(isinstance(date_signature, float)) else ""

            repr_arrete_text= ttr_arr + ttr_rec + d_pub_rec + T_page_arr + v_sign_arr + signs_arr + dt_sign_arr


            arrete={"id_recueil": id_rc,
                    "date_pub_recueil": date_pub,
                    "total_page_recueil":len(pages), # Calcule le nombre de page
                    "titre_arrete": titre_ar,
                    "nbr_arrete": len(liste_num_art) if not(isinstance(liste_num_art, float)) else liste_num_art, # recupère le nombre d'articles par longueur  de la liste  des articles
                    "articles": ''.join(liste_article_arr) if not(isinstance(liste_article_arr, float)) else liste_article_arr, # Fusionne les articles de la liste d'article en un bloc 
                    "liste_articles_arrete": liste_article_arr,
                    "visas": extract_component_arr(pages, interval_p_rc[num_arr], component= "visas"),
                    "repr_arrete_text":repr_arrete_text,
                    "total_page_arrete": total_page_arr,
                    "ville_signature_arrete": ville,
                    "date_signature_arrete": date_signature,
                    "signataire_arrete": signataires}
            liste_objet_arrete.append(arrete)   
            
            print(f"Succesfull reading of arrete n°{num_arr +1 }") # Affichage dans la console pour suivre l'exécution du traitement des arrêtés
            
    df_text= pd.DataFrame(liste_objet_arrete)

    return(df_text)


def pdf_paths_to_df(liste_url_recueil):
    """" Fonction qui tansforme une url-receuil en base de donnée recueil en passant par 
    des proccessus d'océrisation de chaque page et de traitement spécifique d'extraction de donnée et 
    en modélisation de database récapitulant tout cela. Il prend en entrée une liste d'url-recueil et 
    retourne la base de donnée associé aux informations extraite de celle ci"""

    num_url= 0
    df_recueils= pd.DataFrame()

    for url_recueil in liste_url_recueil:
        
        num_url+=1
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'  # Définition d'un user-agent pour effectuer la requête HTTP (pour imiter un navigateur)
        URL_REQ_GET=  requests.get(url_recueil,headers={'User-Agent': user_agent}) # Requête de récupération des donnée issues de l'url notamment le pdf
        
        with tempfile.NamedTemporaryFile(suffix=".pdf",delete= True) as temp_pdf: # Utilise le fichier temporaire avec la fonction tempfile
            print(f"URL {num_url}")  # Affichage dans la console pour suivre l'exécution du traitement du recueil (Océrisation)
            print(url_recueil)

            temp_pdf.write(URL_REQ_GET.content)
            temp_pdf.flush()  # Assure que le fichier est écrit complètement
            
            # Convertit chaque page PDF en chaines de caractère (y compris les pages contenant des images)
            pages = convert_from_path(temp_pdf.name)
            df_receive= pages_to_df(pages)    
        df_recueils= pd.concat([df_recueils, df_receive], ignore_index=True)  # Ajoute les extractions d'un arrêté les uns en dessous des autres

    return df_recueils      