import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_bvpl_news_from_url(response, label):
    soup = BeautifulSoup(response.content, 'html.parser')

    el_title = soup.find('h1', class_='post-title')
    el_content = soup.find('div', class_='post-content')

    if el_title:
        title = el_title.get_text()
    else:
        title = ''
    
    if el_content:
        content = ''
        paragraphs = el_content.find_all('p')
        for paragraph in paragraphs:
            content += '\n' + paragraph.get_text()
    else:
        content = ''

    if title =="" and content=="":
        return None
    
    print(content)

    return  [title, content, label]
