import rich, os
import argparse
from PyPDF2 import PdfReader
from tqdm.auto import tqdm
from rich import traceback

traceback.install(show_locals=True)


parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("textfile")
args = parser.parse_args()

base_folder = "ignore/pdf_data"
pdf_list = os.listdir(args.path)
pdf_list = [os.path.join(base_folder, pdfile) for pdfile in pdf_list]

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

    try:
        for pdf in tqdm(file_list, colour='green'):
            text_extract = pdf2text_file(pdf)
            text_corpus += text_extract

        with open(out_text_file, 'w', encoding='utf-8') as file:
            file.write(text_corpus)

        rich.print(f"PDFs extracted to single text file [bold green]{out_text_file}[/bold green]")
        return text_corpus

    except Exception as e:
        rich.print(f'[bold red] Error in extraction --> {e}')

# if __name__=='main':
merge_pdfs(pdf_list, args.textfile)
