import fitz
import spacy
import nltk
nlp = spacy.load('en_core_web_sm')
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

def preprocessing_text(text):
    doc = nlp(text.lower())
    clean_tokens = []

    for token in doc: 
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha:
            clean_tokens.append(token.text)

    return clean_tokens

def extract_keywords(clean_tokens):
    ...


text = extract_pdf_text(f'{path}/A_High_Performance_Blockchain_Platform_for_Intelligent_Devices.pdf')
# print(text)
text_body, text_references = separate_references(text)
preprocessed_text_body = preprocessing_text(text_body)
print(preprocessed_text_body)









