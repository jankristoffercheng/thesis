import numpy as np

from connection.Connection import Connection
from features.POSFeature import POSFeature

class PostsDAO:
    def addPost(self, username, text, hour, min):
        conn = Connection().getConnection()
        cursor = conn.cursor()
        #get the user id from username
        user_id = '1'
        sql = 'INSERT INTO posts(user_id, text, time) VALUES (' + \
            '\'' + user_id + '\', ' + \
            '\'' + text + '\', ' + \
            '\'' + hour + ':' + min + '\')'
        print(sql)
        #cursor.execute(sql)

    def analyzePosts(self):
        conn = Connection().getConnection()
        cursor = conn.cursor()
        sql = 'SELECT id, text FROM posts'
        cursor.execute(sql)
        row = cursor.fetchone()
        while row is not None:
            id = row['id']
            text = row['text']
            posFeature = POSFeature(text)
            nVerbs = posFeature.nVerbs
            nAdjectives = posFeature.nAdjectives

            cursor2 = conn.cursor()
            sql = 'UPDATE posts SET nVerbs = ' + str(nVerbs) + ', nAdjectives = ' + str(nAdjectives) + ' WHERE id = ' + str(id)
            cursor2.execute(sql)
            row = cursor.fetchone()
            print(id)

    def getTrainingData(self):
        conn = Connection().getConnection()
        cursor = conn.cursor()
        sql = 'SELECT P.nVerbs, P.nAdjectives, U.gender FROM posts P, users U WHERE P.user_id = U.id AND P.nVerbs != 0 AND P.nAdjectives != 0;'
        cursor.execute(sql)
        row = cursor.fetchone()
        dimensions = []
        classes = []
        while row is not None:
            dimension = [row['nVerbs'], row['nAdjectives']]
            dimensions.append(dimension)
            classes.append(row['gender'])
            row = cursor.fetchone()
        return {'Dimensions':dimensions, 'Classes': classes}

    def getTrainingGenderPreferentialData(self):
        conn = Connection().getConnection()
        cursor = conn.cursor()
        sql = 'SELECT P.text, U.gender FROM posts P, users U WHERE P.user_id = U.id;'
        cursor.execute(sql)
        row = cursor.fetchone()
        dimensions = []
        classes = []
        while row is not None:
            dimensions.append(row['text'])
            classes.append(row['gender'])
            row = cursor.fetchone()


        return {'Dimensions':dimensions, 'Classes': classes}


