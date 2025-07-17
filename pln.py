import re
import pandas as pd
import os
import fitz
import spacy
from collections import Counter
from nltk.util import ngrams
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
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
    

def extract_publication_year(text):
    """Extrai o ano de publicação (4 dígitos, começando com 20xx) do texto."""
    match = re.search(r'\b(20[12]\d)\b', text[:5000]) # Busca no início do texto
    return int(match.group(1)) if match else None

def extract_future_work(text_body):
    """Extrai sentenças relacionadas a trabalhos futuros."""
    conclusion_text = get_section_text(text_body, ["Conclusion", "Conclusions", "Future Work"])
    if not conclusion_text:
        conclusion_text = text_body[-int(len(text_body) * 0.20):] # Fallback
    
    keywords = ["future work", "future research", "we plan to", "next step", "promising direction"]
    future_work_sentences = []
    doc = nlp(conclusion_text)
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in keywords):
            future_work_sentences.append(sent.text.strip().replace('\n', ' '))
    return " ".join(future_work_sentences)

# --- Funções de Visualização ---

def visualize_top_terms(df, top_n=20):
    """Gera um gráfico de barras e uma nuvem de palavras para os termos mais comuns."""
    print("\n--- Gerando Visualizações dos Termos Gerais ---")
    
    # 1. Agrega todos os termos de todos os artigos
    global_counter = Counter()
    for term_list in df['termos_frequentes']:
        global_counter.update(dict(term_list))
        
    top_terms = global_counter.most_common(top_n)
    terms, counts = zip(*top_terms)

    # 2. Gráfico de Barras
    plt.figure(figsize=(12, 10))
    sns.barplot(x=list(counts), y=list(terms), palette="viridis")
    plt.title(f'Top {top_n} Termos Mais Citados (Geral)', fontsize=16)
    plt.xlabel('Frequência', fontsize=12)
    plt.ylabel('Termos', fontsize=12)
    plt.tight_layout()
    plt.savefig('output/top_terms_geral.png')
    plt.show()

    # 3. Nuvem de Palavras
    wordcloud = WordCloud(width=1200, height=600, background_color="white", colormap="magma").generate_from_frequencies(global_counter)
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Nuvem de Palavras Geral', fontsize=16)
    plt.savefig('output/wordcloud_geral.png')
    plt.show()

def visualize_techniques(df, techniques):
    """Gera um gráfico de barras da frequência de técnicas específicas."""
    print("\n--- Gerando Visualização das Técnicas Mencionadas ---")
    
    technique_counts = {tech: 0 for tech in techniques}
    for text in df['text_body']:
        for tech in techniques:
            technique_counts[tech] += text.lower().count(tech.lower())
    
    # Filtra técnicas que não foram mencionadas
    mentioned_techs = {k: v for k, v in technique_counts.items() if v > 0}
    if not mentioned_techs:
        print("Nenhuma das técnicas especificadas foi encontrada.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(mentioned_techs.keys()), y=list(mentioned_techs.values()), palette="plasma")
    plt.title('Frequência de Técnicas Mencionadas', fontsize=16)
    plt.xlabel('Técnica', fontsize=12)
    plt.ylabel('Contagem de Menções', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig('output/tecnicas_mencionadas.png')
    plt.show()

def visualize_temporal_evolution(df, terms_to_track):
    """Gera um gráfico de linhas mostrando a evolução de termos ao longo dos anos."""
    print("\n--- Gerando Visualização da Evolução Temporal ---")

    df_filtered = df.dropna(subset=['ano']) # Remove artigos sem ano
    if len(df_filtered) < 2:
        print("Dados insuficientes para análise temporal (necessário mais de 1 ano).")
        return

    # Prepara os dados para o plot
    evolution_data = []
    for index, row in df_filtered.iterrows():
        year = int(row['ano'])
        term_counts = dict(row['termos_frequentes'])
        for term in terms_to_track:
            if term in term_counts:
                evolution_data.append({'ano': year, 'termo': term, 'frequencia': term_counts[term]})
    
    if not evolution_data:
        print("Nenhum dos termos para rastreamento foi encontrado nos artigos.")
        return
        
    evo_df = pd.DataFrame(evolution_data)
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=evo_df, x='ano', y='frequencia', hue='termo', marker='o', lw=2)
    plt.title('Evolução Temporal de Termos-Chave', fontsize=16)
    plt.xlabel('Ano de Publicação', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.legend(title='Termos')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('output/evolucao_temporal.png')
    plt.show()

def visualize_future_work_topics(df):
    """Gera uma nuvem de palavras a partir dos termos encontrados em seções de trabalhos futuros."""
    print("\n--- Gerando Visualização dos Tópicos de Trabalhos Futuros ---")
    
    all_future_text = " ".join(df['trabalhos_futuros'].dropna())
    
    if not all_future_text.strip():
        print("Nenhuma seção de 'Trabalhos Futuros' foi encontrada.")
        return

    future_terms = extract_terms(all_future_text)
    future_counter = Counter(future_terms)
    
    if not future_counter:
        print("Nenhum termo relevante encontrado nas seções de 'Trabalhos Futuros'.")
        return

    wordcloud = WordCloud(width=1200, height=600, background_color="white", colormap="cividis").generate_from_frequencies(future_counter)
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Tópicos proeminentes em "Trabalhos Futuros"', fontsize=16)
    plt.savefig('output/wordcloud_trabalhos_futuros.png')
    plt.show()


def read_pdf_and_extract_information(pdf_path):
    full_text = extract_pdf_text(pdf_path)
    text_body, _ = separate_references(full_text)

    # Definição das pistas para cada tipo de extração
    OBJECTIVE_KEYWORDS = ['objective of this paper', 'this paper aims to', 'we propose', 'the goal of this work is']
    OBJECTIVE_TARGETS = ["the primary goal of this paper is to propose a new framework", "this study aims to evaluate the performance of the existing system", "we present a comprehensive analysis of the security model"]
    OBJECTIVE_SECTIONS = ['Abstract', 'Introduction']

    PROBLEM_KEYWORDS = ['main problem is', 'key challenge is', 'a limitation of']
    PROBLEM_TARGETS = ["the fundamental challenge is the lack of efficiency and scalability", "a major drawback of current approaches is the lack of privacy", "existing solutions suffer from significant performance issues"]
    PROBLEM_SECTIONS = ['Introduction', 'Motivation', 'Problem Statement']

    METHOD_KEYWORDS = ['methodology consists of', 'we used a dataset', 'our approach is based on']
    METHOD_TARGETS = ["our methodology involves a series of experiments on a custom dataset", "we developed a prototype to test the feasibility of the approach", "the study is based on a formal analysis of the protocol"]
    METHOD_SECTIONS = ['Methodology', 'Method', 'Methods', 'Approach', 'Experiments']

    CONTRIBUTION_KEYWORDS = ['our main contribution', 'this study contributes', 'this work provides']
    CONTRIBUTION_TARGETS = ["the key contribution of this work is a novel algorithm for data protection", "our work provides a new framework that significantly improves performance", "we are the first to demonstrate the practicality of this solution"]
    CONTRIBUTION_SECTIONS = ['Conclusion', 'Conclusions', 'Results', 'Introduction']

    # Execução da cascata
    objectives = run_extraction_cascade(text_body, OBJECTIVE_SECTIONS, OBJECTIVE_KEYWORDS, OBJECTIVE_TARGETS, similarity_threshold=0.75, default_msg="Objetivo não encontrado")
    problems = run_extraction_cascade(text_body, PROBLEM_SECTIONS, PROBLEM_KEYWORDS, PROBLEM_TARGETS, similarity_threshold=0.68, default_msg="Problema não encontrado")
    methods = run_extraction_cascade(text_body, METHOD_SECTIONS, METHOD_KEYWORDS, METHOD_TARGETS, similarity_threshold=0.68, default_msg="Metodologia não encontrada")
    contributions = run_extraction_cascade(text_body, CONTRIBUTION_SECTIONS, CONTRIBUTION_KEYWORDS, CONTRIBUTION_TARGETS, similarity_threshold=0.75, default_msg="Contribuição não encontrada")
    
    final_terms = extract_terms(text_body)
    counter_terms = Counter(final_terms)
    most_common_terms = counter_terms.most_common(10)
    
    # *** NOVAS EXTRAÇÕES ADICIONADAS AQUI ***
    publication_year = extract_publication_year(full_text)
    future_work_text = extract_future_work(text_body)
    
    list_information = {
        'arquivo': pdf_path,
        'objetivo': objectives,
        'problema': problems,
        'metodologia': methods,
        'contribuicao': contributions,
        'termos_frequentes': most_common_terms,
        # *** NOVOS DADOS ADICIONADOS AO DICIONÁRIO ***
        'ano': publication_year,
        'trabalhos_futuros': future_work_text,
        'text_body': text_body # Adicionando o corpo do texto para análises futuras
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

    visualize_top_terms(df, top_n=20)
    
    lista_tecnicas_blockchain = ['blockchain', 'smart contract', 'hyperledger fabric', 'solidity', 'pbft', 'byzantine fault tolerance', 'zero-knowledge proof']
    visualize_techniques(df, lista_tecnicas_blockchain)
    
    termos_para_evolucao = ['scalability', 'privacy', 'performance', 'security']
    visualize_temporal_evolution(df, termos_para_evolucao)
    
    visualize_future_work_topics(df)

if __name__ == '__main__':
    main()