import csv

import datetime
import pymysql
import json

import pytz
from dateutil.parser import parse

from features.POSFeature import POSFeature


class ConnectionFactory:
    def getConnectionThesis(self):
        return pymysql.connect(host='localhost', user='root', password='1234', db='thesisdb', charset='utf8mb4',
                           cursorclass=pymysql.cursors.DictCursor, autocommit=True)


def addposts():
    conn = ConnectionFactory().getConnectionThesis()
    conn.set_charset('utf8mb4')
    cursor = conn.cursor()
    cursor.execute('SET NAMES utf8mb4;')
    cursor.execute('SET CHARACTER SET utf8mb4;')
    cursor.execute('SET character_set_connection=utf8mb4;')

    with open('edsthesis_data.json') as data_file:
        data = json.load(data_file)

    philtz = pytz.timezone("Asia/Manila")

    for i in data:

        query = 'Select * from User where Username = %s'
        cursor.execute(query, i['experiment_id'])
        date_object = parse(i['created_at'])
        philver = date_object.astimezone(philtz)

        delta = philtz.localize(datetime.datetime.now()) - philver
        if (delta.days <= 365):
            id = -1
            if (cursor.rowcount > 0):
                row = cursor.fetchone()
                id = int(row['Id'])

                posproc = POSFeature(i['text'])


                try:
                    cursor.execute('INSERT INTO Post(User, Text, Time, POS) VALUES (%s,%s,%s,%s)',
                                   (id, i['text'], philver.strftime("%H:%M"), posproc.sPOS))
                except Exception:
                    print('fuuu')

def addusers(limit=None):
    conn = ConnectionFactory().getConnectionThesis()
    conn.set_charset('utf8mb4')
    cursor = conn.cursor()
    cursor.execute('SET NAMES utf8mb4;')
    cursor.execute('SET CHARACTER SET utf8mb4;')
    cursor.execute('SET character_set_connection=utf8mb4;')

    with open('PersonalInfo_filtered.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')

        ind = 0
        for row in readCSV:
            if ((limit == None or ind < limit) and (row[11] == "PH" and int(row[12].split()[0]) >= 8)):
                sex = 'F'
                if (row[4] == 'male'):
                    sex = 'M'
                cursor.execute('INSERT INTO User(Username, Gender, Birthdate, Source) VALUES (%s,%s,%s,%s)',
                               (row[0], sex, row[3] + "-" + str(row[1]).zfill(2) + "-" + row[2], 'Twitter'))
                ind+=1

def fixjson():
    import fileinput

    with fileinput.FileInput('edsthesis_data.json', inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace('}{', '},{'), end='')



addusers(5)
addposts()