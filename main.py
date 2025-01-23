import time
import warnings
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pytrends.request import TrendReq
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from retrying import retry
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

warnings.simplefilter(action="ignore", category=FutureWarning)
st.set_page_config(page_title="Keyword and Blog Generator", layout="wide")

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    max_output_tokens=2000,
    google_api_key=GOOGLE_API_KEY,
)

pytrends = TrendReq(hl="en-US", tz=360)

@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def fetch_keyword(pytrends, keyword):
    pytrends.build_payload([keyword], timeframe="today 3-m")
    return pytrends.interest_over_time()

def fetch_image_from_pexels(query):
    """Fetch an image URL from Pexels API based on a query."""
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=1"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data["photos"]:
            return data["photos"][0]["src"]["large"]
    return None

def generate_blog_titles(input_word, category, location, age_group):
    """Generate multiple descriptive blog headings based on a topic and other input parameters."""
    prompt = (
        f"Generate 5 descriptive blog titles for a product named '{input_word}' "
        f"in the '{category}' category, targeting customers in the '{location}' region, "
        f"and suitable for the '{age_group}' age group."
    )
    response = model.generate_content(prompt)
    return response.text.strip().split("\n")

def generate_blog_content(prompt):
    """Generate blog content using Gemini API."""
    response = model.generate_content(f"Write a detailed blog about: {prompt}")
    return response.text.strip()

st.title("BloggerAI")
st.sidebar.header("Input Parameters")

input_word = st.sidebar.text_input("Enter Product Name (Blog Topic)")
category = st.sidebar.text_input("Enter Product Category")
location = st.sidebar.text_input("Enter Location")
age_group = st.sidebar.text_input("Enter Age Group")

if input_word and category and location and age_group:
    with st.spinner("Generating Blog Titles..."):
        blog_titles = generate_blog_titles(input_word, category, location, age_group)

    st.subheader("Suggested Blog Titles:")
    for title in blog_titles:
        st.write(title)

    st.write("Analyzing Composite Scores...")
    progress_bar = st.progress(0)

    blog_scores = []
    blog_data = {}

    for idx, blog_title in enumerate(blog_titles):
        keyword_prompt_template = PromptTemplate(
            input_variables=["input", "category", "location", "age_group"],
            template=(
                "For the product '{input}' in the '{category}' category, targeting customers in the '{location}' region and the age group '{age_group}', "
                "generate exactly 5 trending and relevant one-word keywords. "
                "Avoid generic terms and ensure each keyword is directly relevant to '{input}'. "
                "Provide only the one-word keywords, separated by commas."
            ),
        )
        formatted_prompt = keyword_prompt_template.format(
            input=input_word, category=category, location=location, age_group=age_group
        )
        result = llm.invoke(formatted_prompt)
        result_text = result.content if hasattr(result, "content") else result.text
        keywords = [keyword.strip() for keyword in result_text.split(",") if keyword.strip()]
        keywords = keywords[:5]  

        composite_score = 0
        keyword_data = {}
        for keyword in keywords:
            try:
                data = fetch_keyword(pytrends, keyword)
                if data.empty or keyword not in data.columns:
                    continue

                avg_interest = data[keyword].mean()
                rolling_avg = data[keyword].rolling(window=5).mean()
                if rolling_avg.iloc[-5] != 0:
                    short_term_growth_rate = (rolling_avg.iloc[-1] - rolling_avg.iloc[-5]) / rolling_avg.iloc[-5]
                else:
                    short_term_growth_rate = 0
                composite_score += avg_interest * 0.7 + (short_term_growth_rate * 100) * 0.6
                keyword_data[keyword] = {
                    "data": data[keyword],
                    "rolling_avg": rolling_avg,
                    "growth_rate": short_term_growth_rate,
                }
            except Exception:
                continue

        blog_scores.append(composite_score)
        blog_data[blog_title] = {"keywords": keywords, "score": composite_score, "data": keyword_data}

        progress_bar.progress((idx + 1) * (100 // len(blog_titles)))

    st.subheader("Composite Scores for Blog Titles:")
    for title, score in zip(blog_titles, blog_scores):
        st.markdown(f"{title}: <span style='color:green; font-weight:bold;'>{score:.2f}</span>", unsafe_allow_html=True)

    selected_title = st.selectbox("Choose a blog title to proceed:", blog_titles)

    if selected_title:
        if st.button("Generate Blog"):
            final_progress_bar = st.progress(0)

            for i in range(1, 101):
                time.sleep(0.02)  
                final_progress_bar.progress(i)

            image_url = fetch_image_from_pexels(selected_title)
            if image_url:
                st.image(image_url, caption=f"Image for: {selected_title}", use_column_width=True)
            else:
                print("No image found")

            st.write(generate_blog_content(selected_title))

            st.subheader("InsightQ")
            st.write("Analyzing Graphs...")

            graph_progress_bar = st.progress(0)
            total_graphs = 4  

            keyword_data = blog_data[selected_title]["data"]
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

            for keyword, data in keyword_data.items():
                axs[0, 0].plot(data["data"], label=f'{keyword}')
            axs[0, 0].set_title('Interest Over Time')
            axs[0, 0].set_xlabel('Date')
            axs[0, 0].set_ylabel('Interest Level')
            axs[0, 0].legend()
            axs[0, 0].grid(True)
            graph_progress_bar.progress(25)

            for keyword, data in keyword_data.items():
                axs[0, 1].plot(data["data"], label=f'{keyword}')
                peak = data["data"].idxmax()
                trough = data["data"].idxmin()
                axs[0, 1].scatter(peak, data["data"][peak], color='red', label=f'{keyword} Peak')
                axs[0, 1].scatter(trough, data["data"][trough], color='blue', label=f'{keyword} Trough')
            axs[0, 1].set_title('Peaks and Troughs')
            axs[0, 1].set_xlabel('Date')
            axs[0, 1].set_ylabel('Interest Level')
            axs[0, 1].legend()
            axs[0, 1].grid(True)
            graph_progress_bar.progress(50)

            growth_rates = [data["growth_rate"] for keyword, data in keyword_data.items()]
            axs[1, 0].bar(keyword_data.keys(), growth_rates, color='skyblue', edgecolor='black')
            axs[1, 0].set_title('Growth Rate Over Time')
            axs[1, 0].set_xlabel('Keywords')
            axs[1, 0].set_ylabel('Growth Rate')
            axs[1, 0].grid(True)
            graph_progress_bar.progress(75)

            interest_levels = [data["data"].mean() for keyword, data in keyword_data.items()]
            axs[1, 1].hist(interest_levels, bins=10, color='orange', edgecolor='black')
            axs[1, 1].set_title('Distribution of Interest Levels')
            axs[1, 1].set_xlabel('Average Interest Level')
            axs[1, 1].set_ylabel('Frequency')
            axs[1, 1].grid(True)
            graph_progress_bar.progress(100)

            plt.tight_layout()
            st.pyplot(fig)
            

else:
    st.warning("Please fill in all required fields.")
