import numpy as np

from connection.Connection import Connection

class PosDAO:
    def addPOS(self, filipino_pos, english_pos):
        conn = Connection().getConnection()
        cursor = conn.cursor()
        sql = 'INSERT INTO posts(filipino_pos, english_pos) VALUES (' + \
            '\'' + filipino_pos + '\', ' + \
            '\'' + english_pos + '\')'
        print(sql)
        #cursor.execute(sql)


    def getPOSMapping(self):
        conn = Connection().getConnection()
        cursor = conn.cursor()
        sql = 'SELECT filipino_pos, english_pos FROM pos_mapping;'
        cursor.execute(sql)
        row = cursor.fetchone()

        mapDict = {}
        while row is not None:
            mapDict[row['filipino_pos']] = row['english_pos']
            row = cursor.fetchone()
        return mapDict



