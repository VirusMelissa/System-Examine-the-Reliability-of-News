import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_thanhnien_news_from_url(response, label):
    
    soup = BeautifulSoup(response.content, 'html.parser')

    el_title = soup.find('h1', class_='detail-title')
    el_content = soup.find('div', class_='detail-content afcbc-body')

    if el_title:
        title = el_title.find('span', class_='').get_text()
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

