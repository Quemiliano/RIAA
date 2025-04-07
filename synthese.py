from database_creator import *
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
import nltk
nltk.download('punkt') # Diviser le texte en phrases
from transformers import AutoTokenizer
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import seaborn as sns


# Fonction de nettoyage du texte


# Nettoyage et la lemmatisation
def nettoyer_et_lemmatiser(texte):
    """Fonction de néttoyage et de lématisation de texte qui prend en entré un texte et qui retourne un un texte sans les mots vides et les mots lemmatisés"""
    
    mots_vides_personnalises = {"document", "article", "arrêter","arrêté", "recueil", "voir", "considérer", "région", "régional", "bretagne", "monsieur", "madame", "autorisation", "rue", "ar", "ars", "annexe", "général"}

    nlp_fr = spacy.load("fr_core_news_sm") # Modèle français
    texte = texte.lower().replace("l'", "le ")
    texte = texte.lower().replace("'", "e ")
    texte = re.sub(r"[^\w\s]", "", texte)
    texte = re.sub(r'\d+', "", texte) # Suppression de chiffres
    
    doc = nlp_fr(texte)
    # Retrait des mots vides français standard et personnalisés et ponctuation  
    mots_utiles = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token) > 2 and token.lemma_ not in mots_vides_personnalises and not(re.search(r"\n",token.text))] 
    return mots_utiles


def cloud(texte, nb_grams, max_words=100, img_title="wordcloud"):
    
    """Fonction de représentation de nuage de mots qui prend en argument un corpus de texte, une valeurs de ngrams, le nombre de mots maximum à afficher et le titre de l'image du nuage de mots et retourne un objet plt"""

    dico_mots_lemmatise= nettoyer_et_lemmatiser(texte) # traitement du corpus 
    
    if nb_grams== 1:
        ngrams_lemmatise = Counter(dico_mots_lemmatise)  # comptage de la fréquence des mots 
    else:
        ngrams = [tuple([bigram[i] for i in range(0,nb_grams)]) for bigram in nltk.ngrams(dico_mots_lemmatise, nb_grams)]
        ngrams_freq = Counter(ngrams) # comptage de la fréquence des mots
        seuil = 3 # suil d'acceptation de la fréquence des ngrams
        ngrams_lemmatise = {ngram_freq[0] + " " + ngram_freq[1] + (" " + ngram_freq[2] if nb_grams==3 else "") : ngrams_freq[ngram_freq] for ngram_freq in ngrams_freq if ngrams_freq[ngram_freq] >= seuil} # selection des ngrams respectant le seuil

    wordcloud = WordCloud(   # construction du nuage de mots 
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',   # Palette de couleur des mots du nuages de mots
        contour_color='black',  
        contour_width=1.5,
        max_words = max_words
    ).generate_from_frequencies(ngrams_lemmatise)

    # Affichage
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("", fontsize=16)
    plt.savefig(f"images/{img_title}.png")  # Enregistre le graphique dans un fichier  
      
    return plt 


def repr_vectoriel(texte, nb_clust):
    """Fonction de représentation vectorielle , il prend en entrée un texte et un nombre de clusters et retourne un objet plt contenant le nuage de mots"""
    
    clean_texte = nettoyer_et_lemmatiser(texte) # Nettoyage et lemmatisation
    clean_texte_join = " ".join(clean_texte)
    mots = word_tokenize(clean_texte_join) #Tokenisation des mots 

    
    mots_filtres = [mot for mot in mots if mot.isalnum()]  # Filtrage des mots alphanumériques

    # Création des embeddings Word2Vec
    model = Word2Vec([mots_filtres], vector_size=2, window=5, min_count=1, workers=4)  # vector_size=2 pour visualisation

    # Récupération des vecteurs et du vocabulaire des mots
    vecteurs = [model.wv[word] for word in mots_filtres]
    vocabulaire = list(model.wv.index_to_key)

    # Clustering avec KMeans
    kmeans = KMeans(n_clusters= nb_clust, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vecteurs)

    # Organisation des clusters
    clusters = {i: [] for i in range(nb_clust)}
    for word, label in zip(vocabulaire, labels):
        clusters[label].append(word)

    # Trouver les mots les plus proches des centres des clusters
    mots_representatifs = []
    for i in range(nb_clust):
        cluster_center = kmeans.cluster_centers_[i]
        
        # Calcul des distances entre chaque mot du cluster et le centre du cluster
        distances = pairwise_distances_argmin_min([cluster_center], [model.wv[word] for word in clusters[i]])[1]
        mots_representatifs.append(clusters[i][int(distances[0])])  # Conversion de l'indice en entier

    # Visualisation des clusters avec les mots les plus représentatifs
    plt.figure(figsize=(10, 6))
    sns.scatterplot(    
        x=[v[0] for v in vecteurs], 
        y=[v[1] for v in vecteurs], 
        hue=labels, 
        palette="tab10",
        legend="full")

    # Affichage des noms des mots les plus représentatifs sur le graphique
    for i, word in enumerate(mots_representatifs):
        idx = vocabulaire.index(word)
        plt.text(vecteurs[idx][0] + 0.02, vecteurs[idx][1] + 0.02, word, fontsize=12, fontweight='bold')
    # modélisation de l'objet plt
    plt.title("Clusters des mots avec leurs représentations vectorielles")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Clusters")
    plt.savefig(f"images/cluster_mots.png")  # Enregistre le graphique dans un fichier

    return plt

# Fonction pour diviser en sections de 1024 tokens
def diviser_en_sections(tokenizer, phrases, max_tokens=400):
    """Cette fonction prend en entrée un objet de tokenizer implémenté , une phrase et un nombre maximum de token, il retourne uneliste de  sections de 1024 tokens au maximum"""

    sections = []
    section = []
    token_count = 0

    for phrase in phrases:
        token_count_phrase = len(tokenizer.encode(phrase)) # Compter les tokens de la phrase actuelle
        if token_count + token_count_phrase > max_tokens: # Si cette phrase dépasse la limite de tokens, on commence une nouvelle section
            sections.append(" ".join(section))
            section = [phrase]  # Commence une nouvelle section
            token_count = token_count_phrase  # Réinitialiser le compteur de tokens
        else:
            section.append(phrase) # on ajoute les phrase dans la même section 
            token_count += token_count_phrase # On incrémente le compteur
    if section:
        sections.append(" ".join(section))     # Ajouter la dernière section
    
    return sections


def great_summary(texte):
    """Fonction qui prend en entré un texte , il le pré-traite, en faire un résumé et reformule ce résumé avant de le retournée"""
    
    texte= texte.lower()
    texte = re.sub(r'article\s+\d+', '', texte, flags=re.IGNORECASE)  # Supprime "article 1, article 2"
    texte = re.sub(r'\(.*?\)', '', texte)  # Supprime le contenu entre parenthèses
    texte = re.sub(r'vu ', '', texte)  # Supprime le contenu entre parenthèses
    texte = re.sub(r'\s+', ' ', texte).strip()  

    # Assurez-vous d'avoir téléchargé le tokenizer pour le modèle utilisé
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    phrases = nltk.tokenize.sent_tokenize(texte) #Le texte est d'abord divisé en phrases

    # Diviser le texte en sections de 1024 tokens
    sections = diviser_en_sections(tokenizer,phrases= phrases)
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Générer un résumé
    a_traduire= pd.Series(sections)

    resume = lambda t : summarizer(t, max_length=1024, min_length=5, do_sample=True)[0]['summary_text'] # implémentation de la pipeline de résumé
    mon_resume= ''.join(list(a_traduire.map(lambda t: resume(t))))
    reformulateur = pipeline("text2text-generation", model="facebook/bart-large", tokenizer="facebook/bart-large") # implémentation de la pipeline de reformulation
    reformulation = reformulateur(mon_resume, max_length=1024, do_sample=True) # Reformulation
    
    return reformulation[0]['generated_text']

