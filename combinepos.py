from dao.PostsDAO import PostsDAO
from features.POSFeature import POSFeature

postsDAO = PostsDAO()
posFeature = POSFeature()
posFeature.populateMappingDictionary()
posts = postsDAO.getPosts(46598)

for post in posts:
    pos = posFeature.getCombinedPOSTag(post)
    postsDAO.updateCombinedPOS(post.id, pos)