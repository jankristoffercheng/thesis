from bs4 import BeautifulSoup
import requests

class Links:
    def get_keywords(link):
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup.findAll(name='meta', attrs={'name':'keywords'})[0]['content'].split(", ")

    def get_title(link):
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup.title.string



#link = "https://t.co/F5uvLaNUS6"
#print("title:", Links.get_title(link))