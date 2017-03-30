import argparse
import json
import uuid
import requests
from os.path import isfile, join
import pdb

def v(verbose, text):
    if verbose:
        print(text)


server = "https://app.receptiviti.com"
api_key = "58aef3b284e37505c2bc912d"
api_secret = "MxHkthlTnVCT8etCHBf9p8mB9vMM9FMjjdA0PJM9EGk"

class Receptiviti():
    def __init__(self,verbose=False):
        """
        initialise a Receptiviti object
        :type server: str
        :type api_key: str
        :type api_secret: str
        """

        self.server = server
        self.api_key = api_key
        self.api_secret = api_secret
        self.verbose = verbose

    def get_person_id(self, person):
        v(self.verbose, 'getting person: {}'.format(person))
        headers = self._create_headers()
        params = {
            'person_handle': person
        }
        response = requests.get('{}/v2/api/person'.format(self.server), headers=headers, params=params)
        if response.status_code == 200:
            matches = response.json()
            if len(matches) > 0:
                return matches[0]['_id']
        return None

    def _create_headers(self, more_headers={}):
        headers = dict()
        headers.update(more_headers)
        headers['X-API-KEY'] = self.api_key
        headers['X-API-SECRET-KEY'] = self.api_secret
        return headers

    def create_person(self, person):
        v(self.verbose, 'creating person: {}'.format(person))
        headers = self._create_headers({'Content-Type': 'application/json', 'Accept': 'application/json'})
        data = {
            'name': person,
            'person_handle': person,
            'gender': 0
        }
        response = requests.post('{}/v2/api/person'.format(self.server), headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            v(self.verbose, 'Http Response: {}'.format(response))
            raise Exception("Creating person failed!")

        return response.json()['_id']

    def delete_person(self, person_id):
        v(self.verbose, 'deleting person: {}'.format(person_id))
        headers = self._create_headers({'Content-Type': 'application/json', 'Accept': 'application/json'})
        url = '{}/v2/api/person/' + person_id
        response = requests.delete(url.format(self.server), headers=headers)
        if response.status_code != 204:
            v(self.verbose, 'Http Response: {}'.format(response))
            raise Exception("deleting person failed!")

    def add_content(self, person_id, content):
        v(self.verbose, 'add content for {}'.format(person_id))
        headers = self._create_headers({'Content-Type': 'application/json', 'Accept': 'application/json'})
        data = {
            'language_content': content,
            'content_source': 6
        }
        response = requests.post('{}/v2/api/person/{}/contents'.format(self.server, person_id), headers=headers,
                                 data=json.dumps(data))

        if response.status_code != 200:
            raise Exception("Adding content failed!")

        return response.json()['_id']

    def get_profile(self, person_id):
        v(self.verbose, 'get profile for {}'.format(person_id))
        headers = self._create_headers({'Accept': 'application/json'})
        response = requests.get('{}/v2/api/person/{}/profile'.format(self.server, person_id), headers=headers)
        if response.status_code != 200:
            raise Exception("Get profile failed!")
        return response.json()

    def get_communication_recommendation(self, person_name, person_contents):
        person_id = self.get_person_id(person_name)
        if person_id is None:
            person_id = self.create_person(person_name)
        for content in person_contents:
            self.add_content(person_id, content)
        return self.get_profile(person_id)["communication_recommendation"]

def get_liwc(content):
    rtvi = Receptiviti()
    person_name = str(uuid.uuid4()).replace("-","")
    person_id = rtvi.create_person(person_name)
    rtvi.add_content(person_id,content)
    profile = rtvi.get_profile(person_id)
    liwc = profile["liwc_scores"]
    rtvi.delete_person(person_id)

    return liwc

def get_receptivity(content):
    rtvi = Receptiviti()
    person_name = str(uuid.uuid4()).replace("-","")
    person_id = rtvi.create_person(person_name)
    rtvi.add_content(person_id,content)
    profile = rtvi.get_profile(person_id)
    receptiviti = profile["receptiviti_scores"]
    rtvi.delete_person(person_id)

    return receptiviti

def get_all(content):
    rtvi = Receptiviti()
    person_name = str(uuid.uuid4()).replace("-","")
    person_id = rtvi.create_person(person_name)
    rtvi.add_content(person_id,content)
    profile = rtvi.get_profile(person_id)

    return profile
