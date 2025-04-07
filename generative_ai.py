from transformers import pipeline

def conv_agent(question, contexte):
    
    # Charger le pipeline pour BigBird
    qa_pipeline = pipeline(
        "question-answering",
        model="google/bigbird-roberta-base",
        tokenizer="google/bigbird-roberta-base",
        max_seq_len=4096 
        )

    # Obtenir la réponse
    response = qa_pipeline(question=question, context=contexte)
    return(response['answer'])
