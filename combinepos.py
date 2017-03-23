from dao.PostsDAO import PostsDAO
from features.POSFeature import POSFeature

postsDAO = PostsDAO()
posFeature = POSFeature()
posts = postsDAO.getPosts()

for post in posts:
    pos = posFeature.getCombinedPOSTag(post)
    postsDAO.updateCombinedPOS(post.id, pos)