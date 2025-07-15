import fitz
import spacy
from collections import Counter
from nltk.util import ngrams
import nltk
nlp = spacy.load('en_core_web_sm')
path = './pdf'

def extract_pdf_text(pdf_path):
    '''
    Função para extrair o texto de um arquivo PDF
    Args:
        pdf_path: Caminho do arquivo PDF
    Returns:
        text: Texto extraído do PDF
    '''
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def separate_references(text):
    '''
    Função para separar o texto em corpo e referências
    Args:
        text: Texto extraído do PDF
    Returns:
        text_body: Texto do corpo do artigo
        text_references: Texto das referências
    '''
    final_half = text[len(text)//2:]
    position_reference = -1
    
    contador = 0
    references_keywords = ['REFERENCES', 'REFERENCES AND NOTES', 'REFERENCES AND ACKNOWLEDGEMENTS', 'REFERENCES AND ACKNOWLEDGMENTS', 'References', 'References and Notes', 'References and Acknowledgements', 'References and Acknowledgments', 'Bibliography', 'BIBLIOGRAPHY']
    
    for keyword in references_keywords:
        contador += 1
        position = final_half.find(keyword)
        if position != -1:
            position_reference = position + len(text)//2
            break
    
    if position_reference != -1:
        text_body = text[:position_reference]
        text_references = text[position_reference:]
        print(contador)
        return text_body, text_references
    
    else:
        return text, ""

def extract_terms(text_body, text_references):
    '''
    Função para extrair termos do texto
    Args:
        text_body: Texto do corpo do artigo
        text_references: Texto das referências
    Returns:
        final_terms: Lista de termos extraídos
    '''
    final_terms = []

    lemmatized_tokens = tokens_lemmatization(text_body)
    print(lemmatized_tokens)
    two_grams_terms = find_ngrams(text_body, 2)
    print(two_grams_terms)
    three_grams_terms = find_ngrams(text_body, 3)
    print(three_grams_terms)
    
    final_terms.extend(lemmatized_tokens)
    final_terms.extend(two_grams_terms)
    final_terms.extend(three_grams_terms)
    
    return final_terms

def tokens_lemmatization(text_body):
    '''
    Função para lematizar os tokens
    Args:
        text_body: Texto do corpo do artigo
    Returns:
        lemmatized_tokens: Lista de tokens lematizados
    '''
    doc = nlp(text_body.lower())
    lemmatized_tokens = []

    for token in doc: 
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha:
            lemmatized_tokens.append(token.lemma_)

    return lemmatized_tokens

def find_ngrams(text_body, n):
    '''
    Função para encontrar n-gramas
    Args:
        text_body: Texto do corpo do artigo
        n: Tamanho do n-grama
    Returns:
        ngram_filtered: Lista de n-gramas filtrados
    '''
    doc = nlp(text_body.lower())
    tokens = [token.text for token in doc if token.is_alpha]
    ngram_filtered = []

    all_tokens = [token for token in doc if token.is_alpha]

    # Encontrar n-gramas
    ngram = ngrams(all_tokens, n)
    stop_words = nlp.Defaults.stop_words
    
    # Filtrar n-gramas para listar palabras que não começam e não terminam com stop words
    if n == 2:
        for gram in ngram:
            token1, token2 = gram[0], gram[1]
            if token1 not in stop_words and token2 not in stop_words:
                ngram_filtered.append(f'{token1.lemma_} {token2.lemma_}')
    else:
        for gram in ngram:
            token1, token2, token3 = gram[0], gram[1], gram[2]
            if not token1.is_stop and not token3.is_stop:
                ngram_filtered.append(f'{token1.lemma_} {token2.lemma_} {token3.lemma_}')

    return ngram_filtered       


text = extract_pdf_text(f'{path}/A_High_Performance_Blockchain_Platform_for_Intelligent_Devices.pdf')
text_body, text_references = separate_references(text)
final_terms = extract_terms(text_body, text_references)
counter_terms = Counter(final_terms)
most_common_terms = counter_terms.most_common(10)
print(most_common_terms)










