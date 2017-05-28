import numpy as np

from connection.Connection import Connection
from features.POSFeature import POSFeature
from model.Document import Document
from model.Post import Post


class PostsDAO:

    def getPosts(self, id = None):
        conn = Connection().getConnection()
        cursor = conn.cursor()
        sql = 'SELECT Id, Text, EngPOS, FilPOS FROM post'
        if id != None:
            sql += ' WHERE id >= ' + str(id)
        cursor.execute(sql)
        row = cursor.fetchone()
        posts = []
        while row is not None:
            posts.append(Post(row['Id'], row['Text'], row['EngPOS'], row['FilPOS']))
            row = cursor.fetchone()
        return posts

    def updateCombinedPOS(self, id, cmbPOS):
        conn = Connection().getConnection()
        cursor = conn.cursor()
        cursor.execute('Update Post SET CmbPOS = %s WHERE id = %s', (cmbPOS, str(id)))

    def addPost(self, user_id, text, hour, min):
        conn = Connection().getConnection()
        cursor = conn.cursor()
        #get the user id from username
        sql = 'INSERT INTO post(user_id, text, time) VALUES (' + \
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
            sPOS = posFeature.sPOS

            cursor2 = conn.cursor()
            sql = 'UPDATE posts SET nVerbs = ' + str(nVerbs) + ', nAdjectives = ' + str(nAdjectives) +', sPOS = ' + str(sPOS)+' WHERE id = ' + str(id)
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


    def getTrainingPOSData(self):
        conn = Connection().getConnection()
        cursor = conn.cursor()
        sql = 'SELECT P.text, P.sPOS, U.gender FROM posts P, users U WHERE P.user_id = U.id;'
        cursor.execute(sql)
        row = cursor.fetchone()
        dimensions = []
        classes = []
        documentList = []
        while row is not None:
            sPOS = row['sPOS']
            document = Document(row['text'],sPOS)
            dimensions.append(sPOS)
            classes.append(row['gender'])
            documentList.append(document)

            row = cursor.fetchone()

        return {'Dimensions': dimensions, 'Classes': classes, 'Documents': documentList}


