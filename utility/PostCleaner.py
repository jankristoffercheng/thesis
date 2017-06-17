import re
import unicodedata
class PostCleaner:

    def __init__(self):
        try:
            # Wide UCS-4 build
            self.emojiDetector = re.compile(u'['  u'\U0001F300-\U0001F64F'   u'\U0001F680-\U0001F6FF'   u'\u2600-\u26FF\u2700-\u27BF]+',  re.UNICODE)
            self.foreignDetector = re.compile(r'[^\u0000-\u007F]')
            self.acronymDetector = re.compile(r'\s([?.!"](?:\s|$))')
            self.linkCleaner = re.compile(
                r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))',
                flags=re.MULTILINE)

        except re.error:
            # Narrow UCS-2 build
            self.emojiDetector = re.compile(u'(' u'\ud83c[\udf00-\udfff]|'  u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'  u'[\u2600-\u26FF\u2700-\u27BF])+',re.UNICODE)

    def normalizeUnicode(self,postContent):
        return unicodedata.normalize('NFKC', postContent);

    def getEmojis(self, postContent):
        #print("get emoji:",postContent)
        postContent = postContent.encode('unicode_escape')
        postContent = str(postContent, 'unicode_escape')

        for i in range(0, len(postContent)-1):
            if(self.emojiDetector.match(postContent[i]) != None and postContent[i+1] != ' '):
                postContent = postContent[:i]+self.insertSpace(postContent[i:])
        #print("result:",postContent)
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

    def changeLinkToText(self, postContent):
        return self.linkCleaner.sub('URL', postContent, count=0)

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