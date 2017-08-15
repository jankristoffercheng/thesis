from dao.Connection import Connection
from features.POSFeature import POSFeature
from model.Document import Document
from model.Post import Post


class PostsDAO:

    def getAllPost(self):
        conn = Connection().getConnection()
        cursor = conn.cursor()
        sql = 'SELECT text FROM post'
        cursor.execute(sql)
        row = cursor.fetchone()
        posts = []
        while row is not None:
            posts.append(row['text'])
            row = cursor.fetchone()
        return posts

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


