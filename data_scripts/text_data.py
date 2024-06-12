import rich
from PyPDF2 import PdfReader
from tqdm.auto import tqdm 

text_file = 'dhravya_data.txt'


def pdf2text_file(pdf_file: str):
    extract_text = ""
    pdf_read = PdfReader(pdf_file)

    for page_num in tqdm(range(len(pdf_read.pages))):
        page = pdf_read.pages[page_num]
        extract_text += page.extract_text()

    with open(text_file, 'w', encoding='utf-8') as file:
        
        file.write(extract_text)
    
    return extract_text


        
def merge_pdfs(file_list: list, out_text_file: str):
    text_corpus = ''
    
    for pdf in tqdm(file_list):
        text_extract = pdf2text_file(pdf)
        text_corpus += text_extract
    
    with open(text_file, 'wb', encoding='utf-8') as file:
        
        file.write(text_corpus)
    
    return text_corpus