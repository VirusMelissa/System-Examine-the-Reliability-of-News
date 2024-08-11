import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_nhandan_news_from_url(response, label):
    
    soup = BeautifulSoup(response.content, 'html.parser')

    el_title = soup.find('h1', class_='article__title cms-title')
    el_content = soup.find('div', class_='article__body cms-body')

    if el_title:
        title = el_title.get_text()
    else:
        title = ''
    
    if el_content:
        content = ''
        paragraphs = el_content.find_all('p', class_='')
        for paragraph in paragraphs:
            content += '\n' + paragraph.get_text()
    else:
        content = ''

    if title =="" and content=="":
        return None
    

    return  [title, content, label]

