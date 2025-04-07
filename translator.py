import sentencepiece
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import regex as re
def traduction(texte, lang_initiale, lang_cible):
    
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|w{2,4}\.[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+|\b\S*\.fr\b' # suppression des urls car crée des problèmes de traduction quand il est présent
    texte = re.sub(url_pattern, '', texte)
    # Chargement du modèle et tokenizer
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)

    # Configuration des langues
    tokenizer.src_lang = lang_initiale

    # Découper le texte en phrases pour éviter les dépassements
    segments = texte.split('.')
    traductions = []

    for segment in segments:
        if not segment.strip():
            continue
        # Traduction segmentée
        inputs = tokenizer(segment, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.get_lang_id(lang_cible)
        )
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        traductions.append(translated_text)

    return texte, " ".join(traductions)
