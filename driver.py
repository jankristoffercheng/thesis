import nltk
import pymysql
from connection.Connection import Connection
from controller.Controller import Controller
from dao.PostsDAO import PostsDAO
from features.POSFeature import POSFeature
from sklearn.externals import joblib

print("hello world")

# conn = Connection().getConnection()

'''
cursor = conn.cursor()
sql = 'SELECT * FROM users;'
cursor.execute(sql)
row = cursor.fetchone()
while row is not None:
    print(row['username'])
    row = cursor.fetchone()
controller = Controller()
controller.addUser('conan', 'F', '2000', '01', '01', 'Twitter')
'''
#PostsDAO().analyzePosts()
# print(Controller().classifyGenderUsingSVM('He is running. He is a good boy.'))


# joblib.dump(clf, 'prototype.pkl')

clf = Controller().trainGenderPreferential()
#clf = joblib.load('prototype.pkl')
#print(Controller().classifyGenderUsingSVMwithPersist(clf, '@anierlebasi that is what you are... Honey you\'re my golden star.... I know you could make my wish come true. (Napakanta ako)'))
#print(Controller().classifyGenderUsingSVM("@Aryellitaaa Hahaha. Walaaaa! Sige."))
