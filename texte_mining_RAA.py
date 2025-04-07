# [Imports] 
#!pip install -r requirements.txt
import streamlit as stm
import os
import json
from database_creator import *
from translator import *
from synthese import *
from search_engine import *
from generative_ai import *
import time
from streamlit_option_menu import option_menu
from transformers import pipeline
import base64  # Ajout de l'importation de base64
from io import BytesIO


stm.set_page_config(layout="wide")

# Initialisation du compteur de documents traités
if 'document_count' not in stm.session_state:
    stm.session_state.document_count = 0

@stm.cache_data
def OCR(urls):
    """Charge et traite les données des recueils depuis les URLs."""
    stm.session_state.document_count += 1
    return pdf_paths_to_df(urls)


@stm.cache_data
def SEARCH(start_date, end_date):
    return select_url(start_date, end_date)


@stm.cache_data
def Translate(texte, langue_cible):
    """Effectue la traduction du texte."""
    return traduction(texte, 'fr', langue_cible)

@stm.cache_data

def RESUME(texte):
    return great_summary(texte)

def clear_cache_if_needed():
    if stm.session_state.document_count >= 50:
        stm.cache_data.clear()  # Efface le cache
        stm.session_state.document_count = 0  # Réinitialise le compteur

clear_cache_if_needed()

# Charger le fichier JSON
with open('langues.json', 'r', encoding='utf-8') as f:
    langues = json.load(f)


stm.write("""
# Text mining application 📖 
""")

stm.markdown("""
    <style>
        .frame {
            border: 2px solid #4CAF50; /* Couleur du contour (ici un vert) */
            padding: 20px;             /* Espacement interne du cadre */
            margin: 20px 0;           /* Espacement autour du cadre */
            border-radius: 10px;      /* Coins arrondis */
            background-color: #f9f9f9; /* Couleur de fond à l'intérieur du cadre */
        }
    </style>
    <div class="frame">
        <h3>Moteur de recherche des recueils d'actes administratifs (RAA) de la région Bretagne 🔍 </h3>
    </div>
""", unsafe_allow_html=True)
# importer= stm.file_uploader("Importer un recueil CSV", accept_multiple_files=False)
# print(importer)


#Selection
col_select_date_range,texte_de_recherche= stm.columns(2)

# Intervalle de dates de publication

with col_select_date_range:
    tdy= datetime.today()
    date_range = stm.date_input(
        "Intervalle de dates de publication",
        # la plus vielle date : (2019, 1, 12)
        value=[date(tdy.year, tdy.month, 1), tdy.date()],  # Par défaut, aucune date sélectionnée
        help="Indication des périodes de publication des recueils administratifs, avec les dates de début et de fin."
    )
    
    mes_recueils= SEARCH(start_date= date_range[0], end_date= (date_range[1] if len(date_range)== 2 else datetime.today().date()))

    with texte_de_recherche:
        user_text = stm.text_area("Phrase de recherche", "", help= "Affinez et améliorez votre recherche en fournissant des mots-clés ou des phrases spécifiques pour obtenir des résultats plus ciblés et pertinents.")
    
    if len(user_text.replace(" ", ""))> 2:
        list_ref_sommaire_recueil= recup_tuple_ref_sommaire(date_range[0], date_range[1])
        liste_ref= similitude(list_ref_sommaire_recueil, user_text, nbr_affiche= 2)
        #user_text.values= None
        #user_text.empty()
    else:
        liste_ref= mes_recueils.loc[:, "ref_recueil"]

    # Référence du recueil
    select_ref_recueil= stm.selectbox("Référence du recueil", 
                                liste_ref, 
                                index= None,
                                help= "Liste des 3 recueils les plus pertinents en fonction de l'analyse de la phrase de recherche si celle-ci est renseignée, ou affiche tous les recueils si aucune phrase n'est fournie")



if select_ref_recueil :
    my_url= list(mes_recueils[mes_recueils["url"].map(lambda u: True if select_ref_recueil in u else False)]["url"])
    recueil_df = OCR(my_url)
    liste_arrete= list(recueil_df.titre_arrete)
    succes_upload= stm.success("Les données ont été chargées avec succès !", icon="✅")
    time.sleep(1)
    succes_upload.empty()
else:
    recueil_df= None
    liste_arrete= []


with stm.sidebar:
    selected_tab = option_menu("Menu principal", ["Visualiseur de recueil", "Synthèse de document","Traducteur linguistique", "Agent conversationnelle"],
                               icons=["","book", "pencil", "robot"], menu_icon="cast", default_index=0)

# # Par défaut, un onglet est sélectionné
# if 'selected_tab' not in locals():
#     selected_tab = "Visualiseur de recueil"

if selected_tab == "Visualiseur de recueil" and select_ref_recueil :
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Accept': 'application/pdf',
    'Referer': 'https://www.prefectures-regions.gouv.fr/'  # Référent pour simuler une navigation
    }
    try:
        response = requests.get(
            f"{my_url[0]}",
            headers=headers,
            timeout=120,
            allow_redirects=True , verify=False  # Augmenter le délai d'attente
        )
        response.raise_for_status()  # Vérifie les erreurs HTTP

        # Vérification de si le contenu est bien un PDF
        if response.headers['Content-Type'] == 'application/pdf':
            # Convertir la réponse en bytes pour Streamlit
            pdf_data = response.content
            stm.markdown("---")

            # Affichage le PDF dans un iframe en utilisant un composant HTML
            stm.write("### Prévisualisation du PDF :")
            
            # Convertir les bytes en base64
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            pdf_url = f"data:application/pdf;base64,{pdf_base64}"
            
            stm.write(f"Ouvrir le fichier à partir du navigateur: {my_url[0]}")
            stm.markdown("---")
            stm.markdown(f'<iframe src="{pdf_url}" width="1000" height="1000"></iframe>', unsafe_allow_html=True)
            
    except requests.exceptions.RequestException as e:
        stm.error(f"Erreur lors du téléchargement du fichier : {e}")

# Affichage du contenu en fonction de l'onglet sélectionné
if selected_tab == "Traducteur linguistique":

    stm.markdown("---")

    stm.markdown("""
        <style>
            .frame {
                border: 2px solid #4CAF50; /* Couleur du contour (ici un vert) */
                padding: 20px;             /* Espacement interne du cadre */
                margin: 20px 0;           /* Espacement autour du cadre */
                border-radius: 10px;      /* Coins arrondis */
                background-color: #f9f9f9; /* Couleur de fond à l'intérieur du cadre */
            }
        </style>
        <div class="frame">
            <h3>Traducteur 🔡</h3>
        </div>
    """, unsafe_allow_html=True)

    # Filtre sur la partie à traduire
    col_trad_arrete, col_select_filtre, col_select_langue  = stm.columns(3)
    with col_trad_arrete:
        select_arrete= stm.selectbox("Arrêté à traduire", 
                                liste_arrete , 
                                index= None)

    with col_select_filtre:
        select_filtre= stm.selectbox("Partie à traduire", 
                                    ["Tous","Articles", "Visas"], 
                                    index= None)

    with col_select_langue:
        mes_langues= list(langues.keys())
        select_langue=stm.selectbox("Langue cible", 
                                    mes_langues, 
                                    index= mes_langues.index("Anglais"))


    if select_arrete is not None and select_filtre== "Tous" and liste_arrete!= []:
        articles= recueil_df[recueil_df.titre_arrete== select_arrete]["articles"]
        articles= '' if list(pd.isna(articles))[0] else list(articles)[0]
        
        visas= recueil_df[recueil_df.titre_arrete== select_arrete]["visas"]
        visas= '' if list(pd.isna(visas))[0] else list(visas)[0]
        
        texte= visas +"\n" + articles 

    elif select_arrete is not None and select_filtre is not None and liste_arrete!= []: 
        texte= recueil_df[recueil_df.titre_arrete== select_arrete][select_filtre.lower()]
        texte= f'Aucun {select_filtre.lower()} à traduire' if list(pd.isna(texte))[0] else list(texte)[0]
        
    else:
        texte= ""

    valide_trad= stm.button("Traduire", type="primary")
    
    if valide_trad:
        ma_traduction= ("_", "_") if texte== "" else Translate(texte, langues[select_langue])


        col_text_init, col_text_cible= stm.columns(2)

        with col_text_init: 
            stm.markdown(f"""
                <div class="frame">
                    <h4>Texte initiale</h4>
                    <p>
                    {ma_traduction[0]}
                    </p>
                </div>
            """, unsafe_allow_html=True)

        with col_text_cible: 
            stm.markdown(f"""
                <div class="frame">
                    <h4>Texte traduit</h4>
                    <p>
                        {ma_traduction[1]}
                    </p>
                    
                </div>
            """, unsafe_allow_html=True)

elif selected_tab == "Synthèse de document":

    stm.markdown("---")

    stm.markdown("""
    <style>
        .frame {
            border: 2px solid #4CAF50; /* Couleur du contour (ici un vert) */
            padding: 20px;             /* Espacement interne du cadre */
            margin: 20px 0;           /* Espacement autour du cadre */
            border-radius: 10px;      /* Coins arrondis */
            background-color: #f9f9f9; /* Couleur de fond à l'intérieur du cadre */
        }
    </style>
    <div class="frame">
        <h3>Synthétiseur d'arrêté 📜</h3>
    </div>
""", unsafe_allow_html=True)
    
    select_syn_arrete= stm.selectbox("Arrêté à synthétiser", 
                        liste_arrete , 
                        index= None)

    tab_selection_syn = stm.radio("Choisissez le mode de synthèse", ["Nuage de mots","Clusters de mots", "Générateur de résumé - ßêta"])

        
    if select_syn_arrete is not None and liste_arrete!= []:
        articles_syn= recueil_df[recueil_df.titre_arrete== select_syn_arrete]["articles"]
        articles_syn= '' if list(pd.isna(articles_syn))[0] else articles_syn.iloc[0]
        
        visas_syn= recueil_df[recueil_df.titre_arrete== select_syn_arrete]["visas"]
        visas_syn= '' if list(pd.isna(visas_syn))[0] else visas_syn.iloc[0]

        desc_arr= recueil_df[recueil_df.titre_arrete== select_syn_arrete]["repr_arrete_text"].iloc[0]

        texte_syn= f"{desc_arr}. {visas_syn}. {articles_syn}."  

        if tab_selection_syn == "Nuage de mots":
            
            ngrams_col,total_mots_col= stm.columns(2)

            with ngrams_col:
                ngram_syn = stm.selectbox("Taille du groupe de mots (ngrams)", 
                                    [1,2,3], 
                                    index= 0)
                
            with total_mots_col:
                slider_syn_tot_mot = stm.slider(
                    "Nombre total de mot/groupe de mot à afficher",  
                    min_value=1,        
                    max_value=100,        
                    value= 50,      
                    step=1)

            stm.pyplot(cloud(texte_syn,nb_grams= ngram_syn, max_words= slider_syn_tot_mot))
        
        elif tab_selection_syn == "Clusters de mots":
            nbr_cluster= stm.selectbox("Nombre de clusters", 
                                    range(2,11), 
                                    index= 1)
            clustering= repr_vectoriel(texte_syn, nb_clust= nbr_cluster)
            stm.pyplot(clustering)

        elif tab_selection_syn == "Générateur de résumé - ßêta":
            ma_synthese= RESUME(texte_syn)
            stm.markdown(f"""
                <div class="frame">
                    <h4>Résumé</h4>
                    <p>
                    {ma_synthese}
                    </p>    
                </div>
            """, unsafe_allow_html=True)
            actualise_syn= stm.button("🔄 Rafraîchir")
            if actualise_syn:   ma_synthese= great_summary(texte_syn)
            
elif selected_tab == "Agent conversationnelle":

    # Sélection de l'arrêté
    select_study_arrete = stm.selectbox("Arrêté à étudier", liste_arrete, index=None)
    
    stm.markdown("---")
    stm.header("Génerative AI - ßêta 📝✨")
    

    # Extraction des données associées à l'arrêté
    if select_study_arrete is not None and liste_arrete != []:
        articles_study = recueil_df[recueil_df.titre_arrete == select_study_arrete]["articles"]
        articles_study = '' if articles_study.isna().any() else articles_study.iloc[0]

        visas_study = recueil_df[recueil_df.titre_arrete == select_study_arrete]["visas"]
        visas_study = '' if visas_study.isna().any() else visas_study.iloc[0]

        desc_arr= recueil_df[recueil_df.titre_arrete == select_study_arrete]["repr_arrete_text"].iloc[0]
        
        texte_study= f"{desc_arr}. {visas_study}. {articles_study}." 

    # Initialiser l'historique des conversations
    if "messages" not in stm.session_state:
        stm.session_state.messages = []

    # Colonnes pour la question et la réponse
    quest_col, answer_col = stm.columns(2)

    with quest_col:
        # Zone de saisie pour la question
        question_input = stm.text_area("**Question**", "")

        send_quest_col, rafraichir_hist_col, historique_gen_col = stm.columns(3)
        with send_quest_col:
            send_quest = stm.button("📤 Envoyer")
        with rafraichir_hist_col:
            rafraichir_hist = stm.button("🔄 Rafraîchir")
        with historique_gen_col:
            historique_gen= stm.button("⏱️ Historique")


        # Réinitialiser l'historique
        if rafraichir_hist:
            stm.session_state.messages = []  # Réinitialisation

        # Gestion de l'envoi de la question
        if send_quest:
            if question_input and select_study_arrete:
                # Obtenir une réponse via l'agent conversationnel
                answers = conv_agent(question=question_input, contexte=texte_study)

                # Ajouter question et réponse à l'historique
                stm.session_state.messages.append({"user": question_input, "bot": answers})
            else:
                stm.warning("Veuillez sélectionner un arrêté à étudier")

    with answer_col:
        # Afficher la réponse la plus récente si elle existe
        if stm.session_state.messages:
            latest_message = stm.session_state.messages[-1]
            stm.markdown(f"""
                <div class="frame">
                    <h6>Réponse IA</h6>
                    <p>
                    {latest_message['bot']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    if historique_gen:
        # Afficher l'historique complet des messages
        for msg in stm.session_state.messages:
            stm.markdown(f"**Vous :** {msg['user']}")
            stm.markdown(f"**IA :** {msg['bot']}")


# Run
os.system("streamlit run texte_mining_RAA.py --server.headless true")

