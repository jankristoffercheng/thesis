import nltk
import jpype
from jpype import *
import shutil

from prepareedsthesis import ConnectionFactory

NORMALIZE_IN = 'C:/cygwin/normapi/in'
NORMALIZE_OUT = 'C:/cygwin/normapi/normAPIout'
HPOST_IN = 'C:/cygwin/SMTPOST/in'
HPOST_OUT = ''

#get from db idk how
#remember to remove newlines
def getPosts():
    conn = ConnectionFactory().getConnectionThesis()
    conn.set_charset('utf8mb4')
    cursor = conn.cursor()
    query = 'SELECT Id, Text FROM Post;'
    cursor.execute(query)

    ids = []
    posts = []

    row = cursor.fetchone()
    while row is not None:
        ids.append(row['Id'])
        posts.append(row['Text'])
        row = cursor.fetchone()

    return {'ids': ids, 'posts': posts}

def writePostsToFile(posts, filepath):
    target = open(filepath, 'w')
    target.truncate()
    for i in range(len(posts)):
        target.write(posts[i] + '\n')
    target.close()

def getPostsFromFile(filepath):
    target = open(filepath, 'r')
    lines = target.readlines()
    posts = []
    for i in range(len(lines)):
        posts.append(lines[i].rstrip('\n'))
    return posts

def updatePosts(ids, posts):
    if len(ids) == len(posts):
        conn = ConnectionFactory().getConnectionThesis()
        cursor = conn.cursor()
        for i in range(len(ids)):
            cursor.execute('UPDATE Post SET PAOSDASDAS = ' + posts[i] + ' WHERE id = ' + ids[i])
    else:
        print('Cannot update posts')

#get posts with ids
ids = [5,2,7,4,3]
posts = []
posts.append('USERNAME Di kasi lahat ng bagay kinakailangan na balik tanawin. Move on ren teh XD')
posts.append('USERNAME Magpaka sawi ka sa ulan, dpt may paluha effects.')
posts.append('Dapat cnbi m nlng na wala nang klase para di n q ggwa ng paper. HAHAHAHAHAHAHA.')
posts.append('Nsa kabila ka ba nto lht o naghihintay lng aq sa wla...')
posts.append('Bkt b di aq makagawa.')

for post in posts:
    post = ' '.join(nltk.word_tokenize(post))

#ready file for normaliztation
writePostsToFile(posts, NORMALIZE_IN)

#normalize
jvmPath = jpype.getDefaultJVMPath()
jpype.startJVM(jvmPath, "-Djava.class.path=dependencies/NormAPI.jar;dependencies/RBPOST.jar")
normapi = JPackage("normapi").NormAPI
normapi.normalize_File(NORMALIZE_IN)

#copy output of normalized text as hpost input
shutil.copy2(NORMALIZE_OUT, HPOST_IN)

rbpost = JPackage("rbpost").RBPOST
rbpost.hPOST_File(HPOST_IN, '', '')

jpype.shutdownJVM()