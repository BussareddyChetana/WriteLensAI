import re
import math
import numpy as np
from app.nlp.preprocessor import TextPreprocessor

AI_TRANSITION_WORDS = {
    "furthermore", "moreover", "consequently", "nevertheless", "furthermore",
    "in summary", "in conclusion", "to summarize", "it is important to note",
    "delve", "pivotal", "tapestry", "multifaceted", "paramount", "underscores",
    "fosters", "holistic", "beacon", "catalyst", "realm", "crucial role"
}

# Contractions are common in natural human writing (emails, reviews, diary
# entries) and comparatively rare in typical AI-generated prose, which tends
# toward more formal, fully-spelled-out phrasing. This is a topic-independent
# signal, unlike single buzzwords, so it generalizes better across domains.
CONTRACTION_PATTERN = re.compile(
    r"\b\w+'(?:t|re|ve|ll|d|s|m)\b", re.IGNORECASE
)

FIRST_PERSON_PATTERN = re.compile(
    r"\b(i|i'm|i've|i'll|i'd|me|my|mine|we|us|our|ours)\b", re.IGNORECASE
)


class StylometricAnalyzer:
    @staticmethod
    def extract_features(text: str) -> dict:
        """
        Calculates all writing style and statistical features required for WriteLens AI analysis.
        """
        if not text or not text.strip():
            return {
                "word_count": 0,
                "char_count": 0,
                "sentence_count": 0,
                "paragraph_count": 0,
                "reading_time_seconds": 0,
                "reading_time_formatted": "0 sec",
                "avg_sentence_length": 0.0,
                "avg_word_length": 0.0,
                "vocab_richness": 0.0,
                "lexical_diversity": 0.0,
                "type_token_ratio": 0.0,
                "punctuation_density": 0.0,
                "sentence_length_variance": 0.0,
                "sentence_length_std": 0.0,
                "ai_transition_count": 0,
                "ai_transition_density": 0.0,
                "contraction_density": 0.0,
                "first_person_density": 0.0,
                "long_word_ratio": 0.0,
                "punctuation_variety": 0.0
            }

        paragraphs = [p for p in text.split('\n') if p.strip()]
        paragraph_count = max(1, len(paragraphs))
        
        sentences = TextPreprocessor.split_sentences(text)
        sentence_count = max(1, len(sentences))
        
        # Word extraction
        words = re.findall(r'\b[A-Za-z0-9\'-]+\b', text)
        word_count = len(words)
        char_count = len(text)
        
        if word_count == 0:
            word_count = 1  # prevent division by zero
            
        lower_words = [w.lower() for w in words]
        unique_words = set(lower_words)
        
        # Average lengths
        avg_sentence_length = round(word_count / sentence_count, 2)
        avg_word_length = round(sum(len(w) for w in words) / word_count, 2)
        
        # Vocabulary richness & Type-Token Ratio
        type_token_ratio = round(len(unique_words) / word_count, 4)
        vocab_richness = round((len(unique_words) / math.sqrt(word_count * 2)) if word_count > 0 else 0, 4)
        lexical_diversity = type_token_ratio  # synonym as per requirement
        
        # Punctuation density
        punct_count = len(re.findall(r'[.,!?;:\"\'()-]', text))
        punctuation_density = round(punct_count / max(1, char_count), 4)
        
        # Sentence lengths array for burstiness/variance calculation
        sentence_lengths = []
        for sent in sentences:
            sent_words = re.findall(r'\b[A-Za-z0-9\'-]+\b', sent)
            sentence_lengths.append(len(sent_words))
            
        if sentence_lengths:
            sentence_length_variance = round(float(np.var(sentence_lengths)), 2)
            sentence_length_std = round(float(np.std(sentence_lengths)), 2)
        else:
            sentence_length_variance = 0.0
            sentence_length_std = 0.0
            
        # AI transition word density
        transition_matches = 0
        text_lower = text.lower()
        for phrase in AI_TRANSITION_WORDS:
            transition_matches += len(re.findall(r'\b' + re.escape(phrase) + r'\b', text_lower))
            
        ai_transition_density = round((transition_matches / sentence_count), 4)

        # Contraction density: human writing (casual/conversational) uses
        # contractions far more often than typical AI-generated prose.
        contraction_matches = len(CONTRACTION_PATTERN.findall(text))
        contraction_density = round(contraction_matches / word_count, 4)

        # First-person pronoun density: personal narrative/opinion writing
        # (diaries, reviews, emails) skews human; AI essay-style text tends
        # to stay more impersonal/third-person even on personal topics.
        first_person_matches = len(FIRST_PERSON_PATTERN.findall(text))
        first_person_density = round(first_person_matches / word_count, 4)

        # Long word ratio: proportion of words with 7+ letters. AI text
        # tends to lean on more elevated/formal vocabulary consistently,
        # while human writing mixes in more short, plain words.
        long_words = [w for w in lower_words if len(w) >= 7]
        long_word_ratio = round(len(long_words) / word_count, 4)

        # Punctuation variety: count of distinct punctuation marks used.
        # Human writing tends to use a wider, less predictable mix
        # (dashes, ellipses, exclamation points) than typical AI prose.
        punct_chars_used = set(re.findall(r'[.,!?;:\"\'()\-]', text))
        punctuation_variety = len(punct_chars_used)
        
        # Reading time (200 words per minute average reading speed)
        reading_time_minutes = word_count / 200.0
        reading_time_seconds = math.ceil(reading_time_minutes * 60)
        if reading_time_seconds < 60:
            reading_time_formatted = f"{reading_time_seconds} sec"
        else:
            mins = reading_time_seconds // 60
            secs = reading_time_seconds % 60
            reading_time_formatted = f"{mins} min {secs} sec" if secs else f"{mins} min"

        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "reading_time_seconds": reading_time_seconds,
            "reading_time_formatted": reading_time_formatted,
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "vocab_richness": vocab_richness,
            "lexical_diversity": lexical_diversity,
            "type_token_ratio": type_token_ratio,
            "punctuation_density": punctuation_density,
            "sentence_length_variance": sentence_length_variance,
            "sentence_length_std": sentence_length_std,
            "ai_transition_count": transition_matches,
            "ai_transition_density": ai_transition_density,
            "contraction_density": contraction_density,
            "first_person_density": first_person_density,
            "long_word_ratio": long_word_ratio,
            "punctuation_variety": punctuation_variety
        }

    @staticmethod
    def get_feature_vector(text: str) -> list:
        """
        Returns a normalized numerical feature vector for machine learning pipelines.
        """
        stats = StylometricAnalyzer.extract_features(text)
        return [
            stats["avg_sentence_length"],
            stats["avg_word_length"],
            stats["type_token_ratio"],
            stats["punctuation_density"],
            stats["sentence_length_std"],
            stats["ai_transition_density"],
            stats["contraction_density"],
            stats["first_person_density"],
            stats["long_word_ratio"],
            stats["punctuation_variety"]
        ]
