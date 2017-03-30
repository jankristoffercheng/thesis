import re

class PostCleaner:

    def __init__(self):
        try:
            # Wide UCS-4 build
            self.emojiDetector = re.compile(u'['  u'\U0001F300-\U0001F64F'   u'\U0001F680-\U0001F6FF'   u'\u2600-\u26FF\u2700-\u27BF]+',  re.UNICODE)
            self.foreignDetector = re.compile(r'[^\u0000-\u007F]')
            self.acronymDetector = re.compile(r'\s([?.!"](?:\s|$))')
        except re.error:
            # Narrow UCS-2 build
            self.emojiDetector = re.compile(u'(' u'\ud83c[\udf00-\udfff]|'  u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'  u'[\u2600-\u26FF\u2700-\u27BF])+',re.UNICODE)

    def getEmojis(self, postContent):
        print("get emoji:",postContent)
        postContent = postContent.encode('unicode_escape')
        postContent = str(postContent, 'unicode_escape')

        for i in range(0, len(postContent)-1):
            if(self.emojiDetector.match(postContent[i]) != None and postContent[i+1] != ' '):
                postContent = postContent[:i]+self.insertSpace(postContent[i:])
        print("result:",postContent)
        return self.emojiDetector.findall(postContent)

    def insertSpace(self,postContent):

        stop = False
        i = 0
        while(stop == False):
            if(postContent[i] != ' '):
                if (self.emojiDetector.match(postContent[i]) != None and postContent[i+1] != ' ' or
                    self.emojiDetector.match(postContent[i]) == None and self.emojiDetector.match(postContent[i+1]) != None):
                    postContent = postContent[:i+1] + ' ' + postContent[i+1:]
            if(i == len(postContent)-2):
                stop =True
            i+=1
        return postContent

    def removeEmojis(self, postContent):
        postContent = postContent.encode('unicode_escape')
        postContent = str(postContent,'unicode_escape')

        return self.emojiDetector.sub('', postContent, count=0)

    def changeEmojisToText(self, postContent):
        return self.emojiDetector.sub('EMOJI ', postContent, count=0)

    def changeForeignToText(self, postContent):
        return self.foreignDetector.sub('FOREIGN ', postContent,count=0)

    def fixAcronymSpaces(self, postContent):
        return self.acronymDetector.sub(r'\1', postContent, count=0)