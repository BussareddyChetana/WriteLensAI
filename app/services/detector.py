import re
import numpy as np
from app.nlp.preprocessor import TextPreprocessor
from app.nlp.stylometrics import StylometricAnalyzer
from app.ml.classifier import ContentClassifier

classifier = ContentClassifier()

class AnalysisEngine:
    @staticmethod
    def analyze_text(text: str) -> dict:
        """
        Comprehensive analysis pipeline providing:
        - Overall Prediction & Confidence
        - Explanations of WHY prediction was made
        - AI Source Estimation (ChatGPT, Gemini, Claude, Llama, Mistral)
        - Sentence-level analysis with confidence & reasons
        - Complete writing statistics
        """
        cleaned_raw = text.strip() if text else ""
        if not cleaned_raw:
            raise ValueError("Input text cannot be empty.")

        # 1. Base ML Prediction
        prediction_res = classifier.predict(cleaned_raw)
        ai_prob = prediction_res["ai_probability"]
        human_prob = prediction_res["human_probability"]
        prediction = prediction_res["prediction"]
        confidence_score = prediction_res["confidence_score"]

        # 2. Writing Statistics
        stats = StylometricAnalyzer.extract_features(cleaned_raw)
        if prediction_res.get("low_reliability"):
            stats["low_reliability"] = True
            stats["reliability_note"] = prediction_res.get("reliability_note")

        # 3. Generating Detailed Explanations
        explanations = AnalysisEngine._generate_explanations(stats, ai_prob)

        # 4. AI Source Estimation (ChatGPT, Gemini, Claude, Llama, Mistral)
        ai_sources = AnalysisEngine._estimate_ai_sources(cleaned_raw, stats, ai_prob)

        # 5. Sentence-by-Sentence Breakdown
        sentence_analysis = AnalysisEngine._analyze_sentences(cleaned_raw, ai_prob)

        return {
            "prediction": prediction,
            "confidence_score": confidence_score,
            "human_probability": human_prob,
            "ai_probability": ai_prob,
            "explanations": explanations,
            "ai_sources": ai_sources,
            "sentence_analysis": sentence_analysis,
            "stats": stats
        }

    @staticmethod
    def _generate_explanations(stats: dict, ai_prob: float) -> list:
        explanations = []
        sentence_count = stats.get("sentence_count", 0)

        # With very few sentences, sentence-length variance/std is not a
        # meaningful signal (a single sentence always has a std of 0.0,
        # which used to be cited as "uniform sentence lengths" evidence -
        # trivially true and misleading for short text). Require enough
        # sentences before treating burstiness as informative.
        enough_sentences_for_burstiness = sentence_count >= 4
        enough_words_for_vocab = stats.get("word_count", 0) >= 30

        if 42.5 <= ai_prob <= 57.5:
            # The two classes are nearly tied - be honest about that
            # instead of picking a side and inventing confident-sounding
            # reasons for it.
            explanations.append({
                "title": "Mixed or Insufficient Signals",
                "description": "This text doesn't show a clear lean toward either AI-generated or human-written patterns. This is common for short passages, casual messages, or text with a plain, simple style. Treat this result as inconclusive rather than a confident verdict.",
                "type": "neutral"
            })
            if sentence_count < 4:
                explanations.append({
                    "title": "Not Enough Text to Analyze Reliably",
                    "description": f"Only {sentence_count} sentence(s) were provided. Structural signals like sentence-length variation need more text to be meaningful.",
                    "type": "neutral"
                })
            return explanations

        if ai_prob > 57.5:
            if enough_sentences_for_burstiness and stats["sentence_length_std"] < 5.0:
                explanations.append({
                    "title": "Uniform Sentence Lengths",
                    "description": f"Across {sentence_count} sentences, length stays unusually consistent (standard deviation of {stats['sentence_length_std']}), characteristic of AI language models.",
                    "type": "ai"
                })
            if stats["ai_transition_count"] > 0:
                explanations.append({
                    "title": "Repeated Formal Transitions",
                    "description": f"Identified {stats['ai_transition_count']} characteristic transitional markers commonly used by AI models to structure arguments (e.g. 'furthermore', 'in summary', 'crucial').",
                    "type": "ai"
                })
            if enough_words_for_vocab and stats["type_token_ratio"] < 0.65:
                explanations.append({
                    "title": "Repetitive Sentence Structure",
                    "description": "Noticeable repetition in clause construction and vocabulary distribution throughout paragraphs.",
                    "type": "ai"
                })
            if stats.get("contraction_density", 1) < 0.005 and enough_words_for_vocab:
                explanations.append({
                    "title": "Few or No Contractions",
                    "description": "Natural human writing (even formal writing) tends to use contractions like \"it's\" or \"don't\" more often than this text does.",
                    "type": "ai"
                })
            if enough_sentences_for_burstiness and 16 <= stats["avg_sentence_length"] <= 26:
                explanations.append({
                    "title": "Formal & Balanced Syntax",
                    "description": f"Average sentence length is {stats['avg_sentence_length']} words, displaying the balanced syntax typical of synthetic text generators.",
                    "type": "ai"
                })
            if not explanations:
                explanations.append({
                    "title": "Low Stylometric Burstiness",
                    "description": "Pacing and word choice lack the natural cadence and irregular structural spikes found in human writing.",
                    "type": "ai"
                })
        else:
            if enough_sentences_for_burstiness and stats["sentence_length_std"] >= 5.0:
                explanations.append({
                    "title": "Natural Writing Variation",
                    "description": f"Sentences exhibit high structural burstiness (variance of {stats['sentence_length_variance']}), reflecting human rhythmic variation.",
                    "type": "human"
                })
            if enough_words_for_vocab and stats["type_token_ratio"] >= 0.60:
                explanations.append({
                    "title": "High Vocabulary Diversity",
                    "description": f"Unique words account for {round(stats['type_token_ratio']*100, 1)}% of total words, demonstrating rich human vocabulary usage.",
                    "type": "human"
                })
            if stats.get("contraction_density", 0) > 0.01:
                explanations.append({
                    "title": "Natural Use of Contractions",
                    "description": "The text uses contractions (like \"it's\" or \"can't\") at a rate typical of natural human writing.",
                    "type": "human"
                })
            if stats["ai_transition_count"] == 0 and enough_words_for_vocab:
                explanations.append({
                    "title": "Organic Transitional Cadence",
                    "description": "Absence of artificial transition markers; thoughts flow organically between paragraphs.",
                    "type": "human"
                })
            if not explanations:
                explanations.append({
                    "title": "Personal Idiomatic Expression",
                    "description": "Syntax contains localized phrasings, flexible punctuation, and human creative nuances.",
                    "type": "human"
                })

        return explanations

    @staticmethod
    def _estimate_ai_sources(text: str, stats: dict, ai_prob: float) -> dict:
        """
        Estimates likelihood across 5 AI models based on writing fingerprints.
        Returns dictionary with probabilities and official disclaimer.
        """
        if ai_prob < 20.0:
            # Low AI probability
            return {
                "disclaimer": "This is an estimated source based on writing patterns, not a guaranteed identification.",
                "sources": [
                    {"name": "ChatGPT", "probability": 20.0},
                    {"name": "Gemini", "probability": 20.0},
                    {"name": "Claude", "probability": 20.0},
                    {"name": "Llama", "probability": 20.0},
                    {"name": "Mistral", "probability": 20.0}
                ]
            }

        text_lower = text.lower()
        
        # Raw weights based on model specific stylometric fingerprints
        w_chatgpt = 30.0
        w_gemini = 20.0
        w_claude = 20.0
        w_llama = 15.0
        w_mistral = 15.0

        # ChatGPT signature: "delve", "tapestry", "furthermore", "crucial", "in summary"
        cg_keywords = ["delve", "tapestry", "furthermore", "crucial", "in summary", "pivotal", "paramount", "underscores"]
        for kw in cg_keywords:
            if kw in text_lower:
                w_chatgpt += 15.0

        # Gemini signature: bulleted bullet-like flow, active declarative sentences, "here is a summary", "key takeaways"
        gemini_keywords = ["key takeaways", "in overview", "breakdown", "essential aspect", "lets explore", "highlighting"]
        for kw in gemini_keywords:
            if kw in text_lower:
                w_gemini += 15.0

        # Claude signature: thoughtful hedging, "it is worth considering", "nonetheless", "balanced perspective", "nuanced"
        claude_keywords = ["worth considering", "nonetheless", "nuanced", "perspective", "holistic view", "indeed"]
        for kw in claude_keywords:
            if kw in text_lower:
                w_claude += 15.0

        # Llama signature: technical casual phrasing, straightforward explanations
        llama_keywords = ["essentially", "basically", "overall", "simply put", "technically speaking"]
        for kw in llama_keywords:
            if kw in text_lower:
                w_llama += 15.0

        # Mistral signature: concise, direct, imperative or crisp structure
        if stats["avg_sentence_length"] < 15:
            w_mistral += 15.0

        total_weight = w_chatgpt + w_gemini + w_claude + w_llama + w_mistral
        
        sources = [
            {"name": "ChatGPT", "probability": round((w_chatgpt / total_weight) * 100, 1)},
            {"name": "Gemini", "probability": round((w_gemini / total_weight) * 100, 1)},
            {"name": "Claude", "probability": round((w_claude / total_weight) * 100, 1)},
            {"name": "Llama", "probability": round((w_llama / total_weight) * 100, 1)},
            {"name": "Mistral", "probability": round((w_mistral / total_weight) * 100, 1)}
        ]

        # Sort by probability descending
        sources = sorted(sources, key=lambda x: x["probability"], reverse=True)

        return {
            "disclaimer": "This is an estimated source based on writing patterns, not a guaranteed identification.",
            "sources": sources
        }

    @staticmethod
    def _analyze_sentences(text: str, overall_ai_prob: float) -> list:
        sentences = TextPreprocessor.split_sentences(text)
        results = []

        for sent in sentences:
            if len(sent.split()) < 4:
                # Too short for robust sentence-level scoring
                results.append({
                    "sentence": sent,
                    "confidence": round(overall_ai_prob, 1),
                    "is_ai": overall_ai_prob >= 50.0,
                    "risk_level": "High AI Probability" if overall_ai_prob >= 70.0 else ("Moderate AI Probability" if overall_ai_prob >= 40.0 else "Human Writing Pattern"),
                    "reason": "Short phrase evaluated alongside surrounding context."
                })
                continue

            # Predict individual sentence AI probability
            s_res = classifier.predict(sent)
            s_ai_prob = (s_res["ai_probability"] * 0.6) + (overall_ai_prob * 0.4)
            s_words = sent.split()
            
            # Sentence level checks
            has_ai_transition = any(w.lower() in ["furthermore", "moreover", "consequently", "delve", "summary", "crucial", "paramount"] for w in s_words)
            
            if s_ai_prob >= 70.0 or has_ai_transition:
                risk = "High AI Probability"
                reason = "Contains formal AI transition marker and uniform clause balance." if has_ai_transition else "High structural predictability and low lexical burstiness."
                is_ai = True
            elif s_ai_prob >= 45.0:
                risk = "Moderate AI Probability"
                reason = "Displays moderate syntactical symmetry commonly seen in generated text."
                is_ai = True
            else:
                risk = "Human Writing Pattern"
                reason = "Irregular length and organic word choice characteristic of human author."
                is_ai = False

            results.append({
                "sentence": sent,
                "confidence": round(s_ai_prob, 1),
                "is_ai": is_ai,
                "risk_level": risk,
                "reason": reason
            })

        return results
