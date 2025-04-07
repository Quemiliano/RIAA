from sklearn.feature_extraction.text import TfidfVectorizer # ou CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from glob import glob
from nltk.corpus import stopwords
from database_creator import *
import streamlit as stm

def lemmatize_text(text):
    """Fonction pour lemmatiser un texte et retiré les mots vides, idéale si le texte est déja un peu pré-traiter"""
    
    nlp = spacy.load("fr_core_news_md")
    doc = nlp(text) # Transformation du texte en objet nlp
    clean_text=  " ".join(token.lemma_ for token in doc if not token.is_stop) # Lémmatisation + suppression de mots vides

    return clean_text

def vectorize(list_corpus):
    """Fonction de vectorisation de corpus, il prend en entrée une liste de corpus et retourne un objet TfidfVectorizer avec des paramètres initiés et la liste des textes vectorisés"""
    sw = stopwords.words('french') # Récupération des mots vides français

    lemmatized_corpus = [lemmatize_text(doc) for doc in list_corpus] # application de la fonction lematize_text à chaque corpus 
    vectorizer = TfidfVectorizer(lowercase=True, stop_words= sw,   # Création d'un TfidfVectorizer
                            ngram_range=(1, 3),
                            use_idf=True, smooth_idf=True, # idf lissé
                            sublinear_tf=False, norm='l2')
    dtm_sommaire_recueil = vectorizer.fit_transform(lemmatized_corpus) #  Les textes sont représentés sont transformés en matrix sparce après avoir apris le vocabulaire et le poids (On as donc dans la matrix des vecteurs pondérés par TF-IDF.)

    return (vectorizer, dtm_sommaire_recueil) # Retourne un tuple (vectorizer, datamatrix des sommaires)

@stm.cache_data
def recup_tuple_ref_sommaire(start_date, end_date):
    """ Fonction dérivée de la fonction extract_sommaire qui prend en entrée l'intervalle de date de publication des recueils et retourne une liste composé pour chaque arrêté d'un tuple composé de l'url-recueil et du sommaire de celui si."""
    
    df_url= select_url(start_date, end_date)  # Récupère un DataFrame contenant les URLs des recueils publiés entre les deux dates spécifiées
    list_tuple_ref_sommaire= []

    # Parcourt les URLs des recueils extraits du DataFrame
    for i, url_recueil in enumerate(list(df_url["url"])):
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36' # Définition d'un user-agent pour effectuer la requête HTTP (pour imiter un navigateur)
        URL_REQ_GET=  requests.get(url_recueil,headers={'User-Agent': user_agent}, timeout=60)
        
        with tempfile.NamedTemporaryFile(suffix=".pdf",delete= True) as temp_pdf: # Utilise le fichier temporaire avec la fonction d'OCR            
            temp_pdf.write(URL_REQ_GET.content) # Écrit le contenu du PDF dans le fichier temporaire
            temp_pdf.flush()  # Assure que le fichier est écrit complètement
            
            pages = convert_from_path(temp_pdf.name) # Convertit les pages du fichier PDF en images à l'aide de la bibliothèque pdf2image
            list_tuple_ref_sommaire.append((list(df_url["ref_recueil"])[i],extract_sommaire(pages)))  # Ajoute un tuple (référence du recueil, sommaire extrait) à la liste
    
    return list_tuple_ref_sommaire

@stm.cache_data
def similitude(list_ref_sommaire_recueil, search_text, nbr_affiche= 3):

    """Fonction qui détermine la similitude entre une phrase et un liste de corpus , 
    il prend en entré un tuple («référence du recueil», «sommaire du recueil») et 
    le nombre de corpus pertinent à afficher et retourne la liste des sommaires 
    les plus similaire à la phrase"""

    list_corpus = [tpl[1] for tpl in list_ref_sommaire_recueil] # récupère la liste de sommaire
    vectorizer, dtm_sommaire_recueil= vectorize(list_corpus)  # Récupère l'objet de vectorisation Tfidf
    search_text_lemmatize= lemmatize_text(search_text)   # lématisse la chaine de recherche
    question_vect= vectorizer.transform([search_text_lemmatize]) # Vectorisation et transformation de la chaine de recherche
    recueil_question_sim = np.squeeze(cosine_similarity(dtm_sommaire_recueil, question_vect)) # Calcul de la similarité cosinuhs entre la chaine de recherche et la datamatrix représentatif des sommaires des recueils
    doc_ref_sort= np.argsort(recueil_question_sim)[::-1] # Trie la position  des indices des recueil du plus pertinent au moins pertinent
    nbr_affiche_retenu= nbr_affiche if len(doc_ref_sort) >= nbr_affiche else len(doc_ref_sort)  # Filtre sur le nombre de référence à afficher
    
    select_receuil_ind= list(doc_ref_sort[:nbr_affiche_retenu])  # récupère la liste des références des recueils
    select_receuil= [list_ref_sommaire_recueil[ind][0] for ind in select_receuil_ind] # selectionne les références à afficher

    return  select_receuil
