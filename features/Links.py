from bs4 import BeautifulSoup
import requests

def get_keywords(link):
    page = requests.get(link)
    soup = BeautifulSoup(page.content, 'html.parser')
    print(soup.findAll(name='meta', attrs={'name':'keywords'})[0]['content'].split(", "))