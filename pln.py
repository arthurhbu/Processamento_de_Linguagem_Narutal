import re
import pandas as pd
import os
import fitz
import spacy
from collections import Counter
from nltk.util import ngrams
import nltk

print("Carregando modelo spaCy (en_core_web_md)...")
nlp = spacy.load('en_core_web_md')
print("Modelo carregado com sucesso.")

path = './pdf'

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def separate_references(text):
    final_half = text[len(text)//2:]
    position_reference = -1
    
    references_keywords = ['REFERENCES', 'REFERENCES AND NOTES', 'REFERENCES AND ACKNOWLEDGEMENTS', 'REFERENCES AND ACKNOWLEDGMENTS', 'References', 'References and Notes', 'References and Acknowledgements', 'References and Acknowledgments', 'Bibliography', 'BIBLIOGRAPHY']
    
    for keyword in references_keywords:
        position = final_half.find(keyword)
        if position != -1:
            position_reference = position + len(text)//2
            break
    
    if position_reference != -1:
        text_body = text[:position_reference]
        text_references = text[position_reference:]
        return text_body, text_references
    else:
        return text, ""

def extract_terms(text_body):
    final_terms = []
    lemmatized_tokens = tokens_lemmatization(text_body)
    two_grams_terms = find_ngrams(text_body, 2)
    three_grams_terms = find_ngrams(text_body, 3)
    
    final_terms.extend(lemmatized_tokens)
    final_terms.extend(two_grams_terms)
    final_terms.extend(three_grams_terms)
    
    return final_terms

def tokens_lemmatization(text_body):
    doc = nlp(text_body.lower())
    lemmatized_tokens = []
    for token in doc: 
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha:
            lemmatized_tokens.append(token.lemma_)
    return lemmatized_tokens

def find_ngrams(text_body, n):
    doc = nlp(text_body.lower())
    ngram_filtered = []
    all_tokens = [token for token in doc]

    ngram = ngrams(all_tokens, n)
    
    if n == 2:
        for gram in ngram:
            token1, token2 = gram[0], gram[1]
            if not token1.is_stop and not token2.is_stop and token1.is_alpha and token2.is_alpha:
                ngram_filtered.append(f'{token1.lemma_} {token2.lemma_}')
    else:
        for gram in ngram:
            token1, token2, token3 = gram[0], gram[1], gram[2]
            if not token1.is_stop and not token3.is_stop and token1.is_alpha and token2.is_alpha and token3.is_alpha:
                ngram_filtered.append(f'{token1.lemma_} {token2.lemma_} {token3.lemma_}')
    return ngram_filtered

def get_section_text(text_body, section_titles, search_from_start=0):
    text_to_search = text_body[search_from_start:]
    
    title_pattern = '|'.join([re.escape(title) for title in section_titles])
    
    section_start_regex = re.compile(r'^\s*(?:[IVX\d]+\.?\s*)?(' + title_pattern + r')\s*$', re.MULTILINE | re.IGNORECASE)
    
    match = section_start_regex.search(text_to_search)
    
    if not match:
        return ""

    start_index = match.end()
    
    next_section_regex = re.compile(r'^\s*(?:[IVX\d]+\.?\s*)([A-Z][a-zA-Z\s-]{3,})\s*$', re.MULTILINE)
    
    next_match = next_section_regex.search(text_to_search, pos=start_index)
    
    if next_match:
        end_index = next_match.start()
        return text_to_search[start_index:end_index].strip()
    else:
        return text_to_search[start_index : start_index + 5000].strip()

def extract_by_keywords(text_section, keywords):
    if not text_section or not text_section.strip():
        return None
    
    doc = nlp(text_section)
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in keywords):
            return sent.text.strip().replace('\n', ' ')
    return None

def extract_by_similarity(text_section, target_phrases, similarity_threshold=0.75):
    if not text_section or not text_section.strip():
        return None

    main_doc = nlp(text_section)
    if not main_doc.has_vector:
        return "Modelo sem vetores, use 'md' ou 'lg'."

    target_docs = [nlp(phrase) for phrase in target_phrases]
    
    best_sentence = None
    highest_similarity = -1.0

    for sent in main_doc.sents:
        if len(sent) < 8 or not sent.has_vector: continue

        current_similarity = max(sent.similarity(target_doc) for target_doc in target_docs)
        
        if current_similarity > highest_similarity:
            highest_similarity = current_similarity
            best_sentence = sent

    if best_sentence and highest_similarity > similarity_threshold:
        return best_sentence.text.strip().replace('\n', ' ')
    
    return None

def run_extraction_cascade(text_body, sections_to_search, keywords, target_phrases, similarity_threshold, default_msg="Não encontrado"):
    section_text = get_section_text(text_body, sections_to_search)
    
    if section_text:
        result_keywords = extract_by_keywords(section_text, keywords)
        if result_keywords:
            return result_keywords
        
        result_similarity = extract_by_similarity(section_text, target_phrases, similarity_threshold)
        if result_similarity:
            return result_similarity

    fallback_result = extract_by_keywords(text_body, keywords)
    if fallback_result:
        return fallback_result

    fallback_similarity = extract_by_similarity(text_body, target_phrases, similarity_threshold)
    if fallback_similarity:
        return fallback_similarity

    return default_msg
    
def read_pdf_and_extract_information(pdf_path):
    full_text = extract_pdf_text(pdf_path)
    text_body, _ = separate_references(full_text)

    OBJECTIVE_KEYWORDS = ['objective of this paper', 'this paper aims to', 'we propose', 'the goal of this work is']
    OBJECTIVE_TARGETS = [
        "the primary goal of this paper is to propose a new framework",
        "this study aims to evaluate the performance of the existing system",
        "we present a comprehensive analysis of the security model"
    ]
    OBJECTIVE_SECTIONS = ['Abstract', 'Introduction']

    PROBLEM_KEYWORDS = ['main problem is', 'key challenge is', 'a limitation of']
    PROBLEM_TARGETS = [
        "the fundamental challenge is the lack of efficiency and scalability",
        "a major drawback of current approaches is the lack of privacy",
        "existing solutions suffer from significant performance issues"
    ]
    PROBLEM_SECTIONS = ['Introduction', 'Motivation', 'Problem Statement']

    METHOD_KEYWORDS = ['methodology consists of', 'we used a dataset', 'our approach is based on']
    METHOD_TARGETS = [
        "our methodology involves a series of experiments on a custom dataset",
        "we developed a prototype to test the feasibility of the approach",
        "the study is based on a formal analysis of the protocol"
    ]
    METHOD_SECTIONS = ['Methodology', 'Method', 'Methods', 'Approach', 'Experiments']

    CONTRIBUTION_KEYWORDS = ['our main contribution', 'this study contributes', 'this work provides']
    CONTRIBUTION_TARGETS = [
        "the key contribution of this work is a novel algorithm for data protection",
        "our work provides a new framework that significantly improves performance",
        "we are the first to demonstrate the practicality of this solution"
    ]
    CONTRIBUTION_SECTIONS = ['Conclusion', 'Conclusions', 'Results', 'Introduction']

    objectives = run_extraction_cascade(text_body, OBJECTIVE_SECTIONS, OBJECTIVE_KEYWORDS, OBJECTIVE_TARGETS, similarity_threshold=0.75, default_msg="Objetivo não encontrado")
    problems = run_extraction_cascade(text_body, PROBLEM_SECTIONS, PROBLEM_KEYWORDS, PROBLEM_TARGETS, similarity_threshold=0.68, default_msg="Problema não encontrado")
    methods = run_extraction_cascade(text_body, METHOD_SECTIONS, METHOD_KEYWORDS, METHOD_TARGETS, similarity_threshold=0.68, default_msg="Metodologia não encontrada")
    contributions = run_extraction_cascade(text_body, CONTRIBUTION_SECTIONS, CONTRIBUTION_KEYWORDS, CONTRIBUTION_TARGETS, similarity_threshold=0.75, default_msg="Contribuição não encontrada")
    
    final_terms = extract_terms(text_body)
    counter_terms = Counter(final_terms)
    most_common_terms = counter_terms.most_common(10)
    
    list_information = {
        'arquivo': pdf_path,
        'objetivo': objectives,
        'problema': problems,
        'metodologia': methods,
        'contribuicao': contributions,
        'termos_frequentes': most_common_terms
    }
    return list_information

def main():
    pdf_path = 'pdf/'
    result_list = []
    
    
    for file in os.listdir(pdf_path):
        if file.endswith('.pdf'):
            print(f"Processando arquivo: {file}...")
            result_list.append(read_pdf_and_extract_information(f'{pdf_path}/{file}'))

    df = pd.DataFrame(result_list)
    df.to_csv('resultados.csv', sep=';', index=False, header=False, encoding='utf-8')
    print("\nProcessamento concluído! Resultados salvos em 'resultados.csv'.")

    with open('resultados.txt', 'w', encoding='utf-8') as f:
        for result in result_list:
            f.write(f"Arquivo: {result['arquivo']}\n")
            f.write(f"Objetivo: {result['objetivo']}\n")
            f.write(f"Problema: {result['problema']}\n")
            f.write(f"Metodologia: {result['metodologia']}\n")
            f.write(f"Contribuição: {result['contribuicao']}\n")
            f.write("Termos frequentes:\n")
            for termo, freq in result['termos_frequentes']:
                f.write(f"  - {termo}: {freq}\n")
            f.write("\n" + "="*40 + "\n\n")

if __name__ == '__main__':
    main()