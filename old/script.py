# hacker_news_ai_analysis.py
from google.cloud import bigquery
import pandas as pd

def get_ai_related_posts():
    # Use your project ID
    project_id = "ai-lab-479123"
    client = bigquery.Client(project=project_id)

    # Optimized query for AI-related stories and comments
    query = """
    WITH ai_stories AS (
      SELECT 
        id, 
        title, 
        text, 
        score, 
        `by` as author, 
        time, 
        type,
        url
      FROM `bigquery-public-data.hacker_news.full`
      WHERE 
        type IN ('story', 'comment')
        AND (
          LOWER(title) LIKE '% ai %'
          OR LOWER(title) LIKE '%artificial intelligence%'
          OR LOWER(title) LIKE '%machine learning%'
          OR LOWER(title) LIKE '%deep learning%'
          OR LOWER(title) LIKE '%neural network%'
          OR LOWER(title) LIKE '%llm%'
          OR LOWER(title) LIKE '%gpt%'
          OR LOWER(title) LIKE '%openai%'
          OR LOWER(title) LIKE '%chatgpt%'
          OR LOWER(text) LIKE '% ai %'
          OR LOWER(text) LIKE '%artificial intelligence%'
          OR LOWER(text) LIKE '%machine learning%'
          OR LOWER(text) LIKE '%deep learning%'
          OR LOWER(text) LIKE '%neural network%'
          OR LOWER(text) LIKE '%llm%'
          OR LOWER(text) LIKE '%gpt%'
          OR LOWER(text) LIKE '%openai%'
          OR LOWER(text) LIKE '%chatgpt%'
        )
        AND time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 YEAR)
    )
    SELECT * FROM ai_stories
    ORDER BY score DESC
    LIMIT 1000
    """

    try:
        print("Running query to get AI-related Hacker News posts...")
        
        # Run the query
        query_job = client.query(query)
        
        # Convert to DataFrame for easier handling
        df = query_job.to_dataframe()
        
        print(f"Successfully retrieved {len(df)} AI-related posts")
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def analyze_data(df):
    if df is None or df.empty:
        print("No data to analyze")
        return
    
    print("\n=== Data Analysis ===")
    print(f"Total posts: {len(df)}")
    print(f"Posts by type:")
    print(df['type'].value_counts())
    
    if 'score' in df.columns:
        print(f"\nAverage score: {df['score'].mean():.2f}")
        print(f"Highest score: {df['score'].max()}")
    
    # Show some sample titles
    print(f"\nSample AI-related titles:")
    for title in df['title'].head(10).dropna():
        print(f"  - {title}")

def save_to_csv(df, filename="hacker_news_ai_posts.csv"):
    if df is not None and not df.empty:
        df.to_csv(filename, index=False)
        print(f"\nData saved to {filename}")
        return filename
    return None

def main():
    print("=== Hacker News AI Posts Collector ===")
    print("Project: ai-lab-479123")
    
    # Get the data
    df = get_ai_related_posts()
    
    # Analyze and display insights
    analyze_data(df)
    
    # Save to CSV for your sentiment analysis
    if df is not None:
        csv_file = save_to_csv(df)
        if csv_file:
            print(f"\nNext steps:")
            print(f"1. Check the file: {csv_file}")
            print(f"2. Use this data for your sentiment analysis")
            print(f"3. The data includes {len(df)} AI-related posts for analysis")

if __name__ == "__main__":
    main()