import nltk
import re
import string
import nltk

# Ensure required NLTK resources are downloaded safely
def init_nltk():
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if 'punkt' in res else f'corpora/{res}')
        except (LookupError, OSError):
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass

init_nltk()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {"the", "a", "an", "and", "or", "but", "if", "because", "as", "what", "which", "this", "that", "these", "those", "then", "just", "so", "than", "such", "both", "through", "about", "against", "between", "into", "throughout", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now"}

try:
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    LEMMATIZER = None

class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans text: lowercase, remove URLs, HTML tags, special chars, emojis, extra whitespace.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Lowercase conversion
        cleaned = text.lower()
        
        # Remove URLs
        cleaned = re.sub(r'https?://\S+|www\.\S+', '', cleaned)
        
        # Remove HTML tags
        cleaned = re.sub(r'<.*?>', '', cleaned)
        
        # Remove Emojis & non-ASCII non-punctuation special chars
        cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)
        
        # Remove special characters keeping basic punctuation space
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"]', ' ', cleaned)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    @staticmethod
    def preprocess_for_ml(text: str) -> str:
        """
        Full ML preprocessing: clean, tokenize, remove stopwords, lemmatize.
        """
        cleaned = TextPreprocessor.clean_text(text)
        
        # Simple word tokenization
        words = re.findall(r'\b[a-z]{2,}\b', cleaned)
        
        # Stopword removal and Lemmatization
        processed_words = []
        for word in words:
            if word not in STOP_WORDS:
                if LEMMATIZER:
                    try:
                        lemma = LEMMATIZER.lemmatize(word)
                    except Exception:
                        lemma = word
                else:
                    lemma = word
                processed_words.append(lemma)
                
        return " ".join(processed_words)

    @staticmethod
    def split_sentences(text: str) -> list:
        """
        Splits raw text into sentences accurately using regex / NLTK.
        """
        if not text:
            return []
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback regex sentence splitter
            sentences = re.split(r'(?<=[.!?])\s+', text)

        # Both sent_tokenize and the regex fallback above only recognize a
        # sentence boundary when the terminal punctuation is followed by
        # whitespace. Casual/fast typing frequently glues the period
        # straight onto the next word ("...hyderabad.i am..."), which
        # silently merges an entire multi-sentence paragraph into one
        # giant "sentence." That collapse is exactly what makes genuine,
        # casually-typed human writing register as suspiciously uniform
        # (a single-sentence sample has zero length variance), so this
        # second pass explicitly splits on punctuation immediately
        # followed by a letter, with no space required.
        refined = []
        for sent in sentences:
            parts = re.split(r'(?<![A-Z][.!?])(?<=[.!?])(?=[A-Za-z])', sent)
            refined.extend(p for p in parts if p.strip())

        return [s.strip() for s in refined if s.strip()]