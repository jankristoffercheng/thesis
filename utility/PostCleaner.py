import re

class PostCleaner:

    def __init__(self):
        try:
            # Wide UCS-4 build
            self.emoji_detector = re.compile(u'['
                                             u'\U0001F300-\U0001F64F'
                                             u'\U0001F680-\U0001F6FF'
                                             u'\u2600-\u26FF\u2700-\u27BF]+',
                                             re.UNICODE)
        except re.error:
            # Narrow UCS-2 build
            self.emoji_detector = re.compile(u'('
                                             u'\ud83c[\udf00-\udfff]|'
                                             u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
                                             u'[\u2600-\u26FF\u2700-\u27BF])+',
                                             re.UNICODE)

    def removeEmojis(self, postContent):
        postContent = postContent.encode('unicode_escape')
        postContent = str(postContent,'unicode_escape')

        return self.emoji_detector.sub('', post_content, count=0)

    def getEmojis(self, postContent):
        postContent = postContent.encode('unicode_escape')
        postContent = str(postContent, 'unicode_escape')

        for i in range(1, len(postContent)-1):
            print(i)
            if (postContent[i:i + 1].upper() == '\U' and postContent[i - 1] != ' '):
                postContent = postContent[:i] + ' ' + postContent[i:]

        return self.emoji_detector.findall(postContent)