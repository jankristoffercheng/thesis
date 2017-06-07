from bs4 import BeautifulSoup

import requests


page = requests.get("http://lxml.de/")
soup = BeautifulSoup(page.content, 'html.parser')
print(soup.findAll(name='meta', attrs={'name':'keywords'})[0]['content'].split(", "))