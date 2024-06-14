import rich, os, re
import argparse
from PyPDF2 import PdfReader
from tqdm.auto import tqdm
from rich import traceback

traceback.install(show_locals=True) # for beautiful traceback messages

# cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("textfile")
args = parser.parse_args()


# get pdf file paths
base_folder = "pdf_data"
pdf_list = os.listdir(args.path)
pdf_list = [os.path.join(base_folder, pdfile) for pdfile in pdf_list]


def clean_text(text):
    regex_format = r"[^\w\s\d!@#\$%\&'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\[\]\^\_\{\|\}~]"  # Leaves behind onl
    cleaned_text = re.sub(regex_format, '', text) 

    return cleaned_text

def pdf2text_file(pdf_file: str):
    extract_text = ""
    pdf_read = PdfReader(pdf_file)

    for page_num in tqdm(range(len(pdf_read.pages))):
        page = pdf_read.pages[page_num]
        extract_text += page.extract_text()
        extract_text = clean_text(extract_text)

    with open('sample.txt', 'w', encoding='utf-8') as file:
        file.write(extract_text)
    
    return extract_text


def merge_pdfs(file_list: list, out_text_file: str):
    text_corpus = ''

    try:
        for pdf in tqdm(file_list, colour='cyan'): # extract from pdf in queue
            text_extract = pdf2text_file(pdf)
            text_corpus += text_extract # concat text and add space before the next one is added
            text_corpus += '\n  '
            rich.print(f'Extracted [white] {pdf}')
            

        with open(out_text_file, 'w', encoding='utf-8') as file:
            file.write(text_corpus) # write to text file
            
        # success message
        rich.print(f"{len(pdf_list)} research paper PDFs extracted to single text file [bold green]{out_text_file}[/bold green]")
        return text_corpus

    # exception hanndling and colored output
    except Exception as e:
        rich.print(f'[bold red] Error in extraction --> {e}') 


merge_pdfs(pdf_list, args.textfile)
