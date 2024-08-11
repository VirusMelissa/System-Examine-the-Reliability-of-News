import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urlparse
import tldextract
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from DanTriCrawlingData import get_dantri_news_from_url
from util import switchFN

def get_news_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    el_title = soup.find('h1', class_='title-detail')
    el_content = soup.find('article', class_='fck_detail')

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
    

    return  [title, content]


def read_file_excel():
    file_path = 'Danh_data.xlsx'
    data = pd.read_excel(file_path)
    return data

def read_mutil_files_excel(file_paths):
    all_data = []
    for file_path in file_paths:
        data = pd.read_excel(file_path)
        all_data.append(data)
    combined_data = pd.concat(all_data)
    return combined_data


def get_data_from_url():
    news = []
    file_paths = ['reliable/data2.xlsx'] 
    # file_paths = ['reliable/data1.xlsx'] 
    # data_from_excel = read_file_excel()
    data_from_excel = read_mutil_files_excel(file_paths)
    for index, row in data_from_excel.iterrows():
        if row.values[0] != '' and row.values[0] != None:
            print(row, 'row.values[0]')
            session = requests.Session()
            retry = Retry(connect=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)

            response = session.get(row.values[0])
            # response = requests.get(row.values[0])

            soup = BeautifulSoup(response.content, 'html.parser')
            print(row.values[0], 'url')
            extracted = tldextract.extract(row.values[0])
            domain = extracted.domain
            # print(domain, 'domain')
            data_from_url = switchFN(domain, response)
            print(data_from_url, 'data_from_url')
            if  data_from_url is not None:
                news.append(data_from_url)

    df = pd.DataFrame(news, columns=['title', 'content', 'labels'])

    time.sleep(1)


    df.to_csv('data_2.csv', index=False)


# Main funtion here

get_data_from_url()