import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urlparse
import tldextract

# def get_news_from_url(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     print(url)
#     el_title = soup.find('h1', class_='title-detail')
#     el_content = soup.find('article', class_='fck_detail')

#     if el_title:
#         title = el_title.get_text()
#     else:
#         title = ''
    
#     if el_content:
#         content = ''
#         paragraphs = el_content.find_all('p', class_='Normal')
#         for paragraph in paragraphs:
#             content += '\n' + paragraph.get_text()
#     else:
#         content = ''

#     if title =="" and content=="":
#         return None
    

#     return  [title, content, 1]

def get_news_from_url(response, label):
    soup = BeautifulSoup(response.content, 'html.parser')
    
    el_title = soup.find('h1', class_='title-detail')
    el_content = soup.find('article', class_='fck_detail')
    print(el_title, 'el_title')
    if el_title:
        title = el_title.get_text()
    else:
        title = ''
    
    if el_content:
        content = ''
        paragraphs = el_content.find_all('p', class_='Normal')
        for paragraph in paragraphs:
            content += '\n' + paragraph.get_text()
    else:
        content = ''

    if title =="" and content=="":
        return None
    

    return  [title, content, label]


def read_file_excel():
    file_path = 'Danh_data.xlsx'
    data = pd.read_excel(file_path)
    return data


def get_data_from_url():
    news = []
    data_from_excel = read_file_excel()
    for index, row in data_from_excel.iterrows():
        # parsed_url = urlparse(row.values[1])
        # domain = parsed_url.netloc
        extracted = tldextract.extract(row.values[0])
        domain = extracted.domain
        print(domain, 'domain')
        data_from_url = get_news_from_url(row.values[0])
        if  data_from_url is not None:
            news.append(data_from_url)

    df = pd.DataFrame(news, columns=['title', 'content', 'labels'])

    time.sleep(1)


    df.to_csv('data.csv', index=False)


# Main funtion here

# get_data_from_url()