from dao.PostsDAO import PostsDAO
from features.POSFeature import POSFeature
from model.Post import Post

postsDAO = PostsDAO()
posFeature = POSFeature()
posFeature.populateMappingDictionary()
posts = postsDAO.getPosts(529677)

#posts = []
#posts.append(Post('00000417459', 'Kimi No Na Wa. Ang ganda please. Lalo na yung OST.', 'NNP-NNP-NNP-NNP-.-NNP-JJ-NN-.-NNP-TO-VB-NNP-.', 'UNK-NNP-CCP-NNPA-DTC-NNC-NNC-PMP-PMP-CCP-PRO-UNK-PMP'))

for post in posts:
    pos = posFeature.getCombinedPOSTag(post)
    postsDAO.updateCombinedPOS(post.id, pos)