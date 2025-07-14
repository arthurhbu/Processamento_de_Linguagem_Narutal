import fitz
# import spacy

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



text = extract_pdf_text(f'{path}/A_High_Performance_Blockchain_Platform_for_Intelligent_Devices.pdf')
text_body, text_references = separate_references(text)
print(text_references)









