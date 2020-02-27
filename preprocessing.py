import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer

class Preprocessing:

    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    # TODO Discuss levels of preprocessing to apply
    """
        INPUT:      TEXT STRING
        RETURNS:    STRING WITH PREPROCESSING APPLIED
    """
    def preprocess(self, text):
            text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)    #   - Removes URLs
            text = re.sub(r'[^\w\s]', '', text)                                             #   - Removes Punctuation
            text = text.replace(u'\xa0', u' ')                                              #   - Removes Break lines
            text = text.split(' ')                                                          #   - Splits up text by Whitespace
            text = [word for word in text if word not in self.stop_words]                   #   - Removes Stop-Words
            text = [self.ps.stem(word) for word in text]                                    #   - Stems words
            text =  list(filter(lambda a: a != '', text))                                   #   - Removes white space
            return ' '.join(text)                                                           #   - Return remaining words
