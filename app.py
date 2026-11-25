
import streamlit as st
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CINEINTEL | Movie Intelligence",
    page_icon="🎬",
    layout="wide"
)

analyzer = SentimentIntensityAnalyzer()

# ---------------- DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: #10121b;
    color: #f5f5f5;
}
h1, h2, h3 {
    color: #f0c674 !important;
}
div[data-testid="stMetric"] {
    background: #202436;
    padding: 18px;
    border-radius: 12px;
}
.stButton button {
    background: #d6a84b;
    color: #10121b;
    border-radius: 9px;
    font-weight: bold;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- ANALYSIS FUNCTIONS ----------------

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.15:
        sentiment = "POSITIVE"
    elif compound <= -0.15:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL / MIXED"

    return sentiment, scores, compound


def detect_emotion(text):
    words = {
        "Happy": ["happy", "joy", "love", "wonderful", "amazing", "beautiful"],
        "Sad": ["sad", "cry", "emotional", "heartbreaking", "tragic"],
        "Angry": ["angry", "hate", "annoying", "furious", "disgusting"],
        "Excited": ["exciting", "thrilling", "adventure", "fantastic", "awesome"],
        "Disappointed": ["boring", "waste", "disappointing", "poor", "terrible"]
    }

    text_words = re.findall(r"\b\w+\b", text.lower())
    counts = {
        emotion: sum(word in text_words for word in keywords)
        for emotion, keywords in words.items()
    }

    if max(counts.values()) == 0:
        return "Unclear"

    return max(counts, key=counts.get)


def aspect_analysis(text):
    aspects = {
        "Acting": ["acting", "actor", "actress", "performance", "cast"],
        "Story": ["story", "plot", "screenplay", "script", "narrative"],
        "Music": ["music", "song", "soundtrack", "background score"],
        "Direction": ["direction", "director", "directed"],
        "Visuals": ["visual", "cinematography", "effects", "graphics", "scenes"]
    }

    results = {}
    for aspect, keywords in aspects.items():
        matches = [word for word in keywords if word in text.lower()]
        if matches:
            score = analyzer.polarity_scores(text)["compound"]
            if score >= 0.15:
                result = "Positive"
            elif score <= -0.15:
                result = "Negative"
            else:
                result = "Neutral / Mixed"
            results[aspect] = result

    return results


def extract_keywords(text):
    stopwords = {
        "this", "that", "with", "were", "they", "have", "from",
        "very", "movie", "film", "there", "their", "about",
        "would", "could", "should", "just", "really", "what",
        "when", "where", "which", "while", "after", "before",
        "into", "also", "than", "then", "them", "been"
    }

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    filtered = [w for w in words if w not in stopwords]
    return Counter(filtered).most_common(8)


def predict_rating(compound):
    # Approximate rating derived from sentiment score.
    return max(1, min(5, round(3 + 2 * compound)))


def make_summary(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:2]) if sentences else text


# ---------------- USER INTERFACE ----------------

st.markdown("<h1 style='text-align:center;'>🎬 CINEINTEL</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#b8b8c8;'>"
    "ADVANCED MOVIE REVIEW INTELLIGENCE SYSTEM</p>",
    unsafe_allow_html=True
)

st.divider()

st.subheader("🎥 Analyze Your Movie Review")

review = st.text_area(
    "Enter your movie review below:",
    placeholder="Example: The movie was fantastic! The acting was brilliant...",
    height=170
)

analyze_button = st.button("🔍 ANALYZE REVIEW", use_container_width=True)

if analyze_button:
    if len(review.strip().split()) < 3:
        st.warning("Please enter a review containing at least 3 words.")
    else:
        sentiment, scores, compound = analyze_sentiment(review)
        emotion = detect_emotion(review)
        aspects = aspect_analysis(review)
        rating = predict_rating(compound)
        keywords = extract_keywords(review)
        summary = make_summary(review)

        st.divider()
        st.header("📊 Analysis Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Overall Sentiment", sentiment)

        with col2:
            st.metric("Sentiment Score", f"{compound * 100:.1f}/100")

        with col3:
            st.metric("Estimated Star Rating", f"{rating}/5 ⭐")

        st.subheader("😊 Emotion Analysis")
        st.info(f"Detected emotion: **{emotion}**")

        st.subheader("📈 Sentiment Breakdown")

        st.progress(scores["pos"], text=f"Positive: {scores['pos'] * 100:.1f}%")
        st.progress(scores["neu"], text=f"Neutral: {scores['neu'] * 100:.1f}%")
        st.progress(scores["neg"], text=f"Negative: {scores['neg'] * 100:.1f}%")

        st.subheader("🎭 Movie Aspect Analysis")

        if aspects:
            for aspect, result in aspects.items():
                st.write(f"**{aspect}:** {result}")
        else:
            st.write("No specific movie aspects were mentioned.")

        st.subheader("🔑 Important Keywords")

        if keywords:
            st.write(", ".join([word for word, count in keywords]))
        else:
            st.write("No important keywords found.")

        st.subheader("📝 Review Summary")
        st.write(summary)

        st.subheader("🍿 Recommendation Insight")

        if compound >= 0.35:
            st.success("The review is positive. The reviewer may recommend this movie.")
        elif compound <= -0.35:
            st.error("The review is negative. The reviewer may not recommend this movie.")
        else:
            st.warning("The review is mixed or neutral. Recommendation is uncertain.")

        st.caption(
            "Technical note: Sentiment uses VADER's rule-based NLP analyzer. "
            "Emotion and aspect labels use keyword-based heuristics. "
            "Star rating is an approximate sentiment-derived estimate, not a "
            "trained rating prediction. These results are not guaranteed to "
            "represent the reviewer's true opinion."
        )

else:
    st.info("👆 Enter a movie review and click ANALYZE REVIEW to begin.")

st.divider()

st.markdown(
    "<p style='text-align:center;color:#888;'>"
    "CINEINTEL | Machine Learning & NLP College Project</p>",
    unsafe_allow_html=True
)
st.markdown("""
<style>
div[data-baseweb="textarea"] textarea {
    color: #111827 !important;
    background-color: #ffffff !important;
    -webkit-text-fill-color: #111827 !important;
    caret-color: #111827 !important;
}

div[data-baseweb="textarea"] textarea::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.stTextArea textarea,
div[data-testid="stTextArea"] textarea,
div[data-baseweb="textarea"] textarea {
    color: #000000 !important;
    background: #ffffff !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    caret-color: #000000 !important;
    font-size: 18px !important;
}

.stTextArea textarea::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {
    color: #666666 !important;
    -webkit-text-fill-color: #666666 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

