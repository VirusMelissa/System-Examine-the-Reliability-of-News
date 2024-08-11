import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_vietnamnet_news_from_url(response, label):
    soup = BeautifulSoup(response.content, 'html.parser')
    
    el_title = soup.find('h1', class_='content-detail-title')
    el_content = soup.find('div', class_='maincontent main-content')

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

