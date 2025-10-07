import streamlit as st
import numpy as np
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient,model
from vanna.milvus import Milvus_VectorStore
from vanna.openai import OpenAI_Chat
import os


class VannaMilvus(Milvus_VectorStore,OpenAI_Chat):
    def __init__(self, llm_client, config=None):
        Milvus_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=llm_client,config=config)

class EmbeddingWrapper:
    def __init__(self, embedder):
        self.embedder = embedder
    def encode_documents(self, texts):
        result=self.embedder.embed_documents(texts)
        return [np.array(r) for r in result]
    
    def encode_queries(self,texts):
        embeddings=[]
        for text in texts:
            embeddings.append(self.embedder.embed_query(text))
        return embeddings

def get_openai_client():
    client=OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=st.secrets["NVIDIA_API_KEY"],
    )
    return client
def Setup_Vanna_VectorDB():
    llm_client=get_openai_client()

    silicon_embedder=OpenAIEmbeddings(
        model="BAAI/bge-m3",
        base_url="https://api.siliconflow.cn/v1",
        api_key=st.secrets["SILICONFLOW_API_KEY"],
    )
    vanna_embedder=EmbeddingWrapper(silicon_embedder)
    # Define Vector DB client
    milvus_url=st.secrets["MILVUS_URI"]
    if os.path.exists(milvus_url):
        os.remove(milvus_url)
        print(f"Removed existing database file {milvus_url}")
    milvus_client=MilvusClient(milvus_url)

    Config_vanna_VectorDB={
        "model": st.secrets["LLM_MODEL"],
        "milvus_client": milvus_client,
        "embedding_function:":vanna_embedder,
        "n_results":5
    }

    #Initialize VannaMilvus
    vanna=VannaMilvus(llm_client,config=Config_vanna_VectorDB)
    return vanna

def Connect_to_SQLite(VannaDB):
   
    VannaDB.connect_to_sqlite(st.secrets["SQLITE_PATH"])
    return VannaDB

def Train_VannaDB_On_SQLite(VannaDB):
    df_ddl = VannaDB.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
    # Train the model on the DDL data
    for ddl in df_ddl["sql"].to_list():
        VannaDB.train(ddl=ddl)
    doc1="""
    This dataset is used to answer questions about the game trends.
    """
    doc2="""
    The games table contains information about the games.
    The app_id is the unique identifier for the game.This is a primary key.
    The name is the name of the game.
    The release_date is the date the game was released.
    The price is the price of the game. Price in USD, 0.0 if its free.
    The short_description is a brief description of the game.
    The positive is the number of positive reviews or votes.
    The negative is the number of negative reviews or votes.
    The min_owners is the minimum number of owners. Used together with max_owners to get an estimate of the player base.
    The max_owners is the maximum number of owners. Used together with min_owners to get an estimate of the player base.
    The hltb_single is the average playtime of the game. This is an estimate.
    """
    doc3="""
    The categories table contains TECHNICAL FEATURES of the games (技术性分类).
    The app_id is the unique identifier for the game.
    The categories column contains technical features such as:
    - Single-player 
    - Multi-player 
    - Steam Achievements 
    - Full controller support 
    - Steam Cloud 
    - PvP
    - Co-op
    The app_id is a foreign key to the games table.
    """
    doc4="""
    The tags table contains GAME GENRES, STYLES AND THEMES (游戏类型/风格/主题).
    The app_id is the unique identifier for the game.
    The tags column contains game types, styles and themes such as:
    - Indie 
    - Action
    - Puzzle
    - Adventure 
    - RPG 
    - Pixel Graphics 
    - 2D
    - Retro 
    - Horror
    - Strategy
    The tag_frequencies column contains the frequencies/popularity of each tag.
    The app_id is a foreign key to the games table.
    """
    doc5="""
    For date-related queries in the games table:
    - The release_date column is stored as TEXT in 'YYYY-MM-DD' format (e.g., '2013-07-09').
    - To filter by year range, use: WHERE release_date BETWEEN 'YYYY-01-01' AND 'YYYY-12-31'
    - To filter by specific year, use: WHERE release_date >= 'YYYY-01-01' AND release_date < 'YYYY+1-01-01'
    - To extract year from release_date, use: CAST(SUBSTR(release_date, 1, 4) AS INTEGER)
    - Always use ORDER BY with LIMIT when finding "most" or "highest" records.
    - For "most positive feedback", use: ORDER BY positive DESC LIMIT 1
    - Example: To find games released in 2020 with most positive reviews:
      SELECT name, positive FROM games 
      WHERE release_date BETWEEN '2020-01-01' AND '2020-12-31' 
      ORDER BY positive DESC LIMIT 1
    """
    doc7="""
    ===ENHANCED SQL GENERATION RULES===
    
    For COMPLEX multi-step queries, follow this pattern:
    
    【STEP 1: Decompose the Question】
    - Identify: TIME_RANGE, METRICS, GROUPING, FILTERS, COMPARISONS
    
    【STEP 2: SQL Structure】
    Use CTEs for clarity:
    WITH step1_filter AS (...),
        step2_calculate AS (...),
        step3_baseline AS (...)
    SELECT ... FROM step2 JOIN step3
    
    【CRITICAL CHECKS】
    ✓ Division? Use: value * 1.0 / NULLIF(denominator, 0)
    ✓ JOINs? Use: COUNT(DISTINCT primary_key)
    ✓ Specific filters? Apply EARLY in first CTE
    ✓ Growth rate? Handle NULL: CASE WHEN LAG(...) IS NULL OR = 0 THEN NULL ELSE ... END
    ✓ Review rate? Filter: WHERE (positive + negative) > 0 before calculating
    
    【COMMON PATTERNS】
    
    Pattern A - Safe Ratio Calculation:
    CASE WHEN (positive + negative) > 0 
         THEN positive * 1.0 / (positive + negative)
         ELSE NULL END
    
    Pattern B - Year-over-Year Growth:
    CASE WHEN LAG(count) OVER (...) IS NULL OR LAG(count) OVER (...) = 0
         THEN NULL
         ELSE (count - LAG(count) OVER (...)) * 1.0 / LAG(count) OVER (...)
    END
    
    Pattern C - Category vs Overall Comparison:
    -- Step 1: Calculate category metric
    WITH cat_metric AS (SELECT category, AVG(metric) FROM ... GROUP BY category),
    -- Step 2: Calculate overall metric
         overall_metric AS (SELECT AVG(metric) FROM ...)
    -- Step 3: Compare
    SELECT * FROM cat_metric CROSS JOIN overall_metric
    """
    doc8="""
    QUERY COMPLEXITY CLASSIFICATION:
    
    LEVEL 1 - SIMPLE (direct SQL):
    - Single metric from one table
    - Basic WHERE filter
    - Simple aggregation (COUNT, SUM, AVG)
    Example: "How many games were released in 2020?"
    
    LEVEL 2 - MODERATE (1-2 CTEs):
    - JOIN 2-3 tables
    - Basic ratio calculation
    - Single dimension grouping
    Example: "What's the average price of Multi-player games?"
    
    LEVEL 3 - COMPLEX (3-5 CTEs):
    - Time-series analysis
    - Growth rate calculation
    - Multiple groupings
    - One comparison
    Example: "Show year-over-year growth for each category from 2015-2022"
    
    LEVEL 4 - VERY COMPLEX (5+ CTEs):
    - Multi-step calculations (growth + ratio)
    - Cross-dimensional analysis
    - Comparative analysis (category vs overall)
    - Multiple filters and aggregations
    Example: "Analyze category growth 2015-2022, find fastest growing in 2020-2022, 
              compare their review rates with overall average"
    
    For LEVEL 3-4: Use step-by-step CTE approach with comments
    
    """
    doc9="""
COMPLEX SQL ANALYSIS PATTERN FRAMEWORK
    
    When encountering multi-step analytical questions, follow this decomposition pattern:
    
    【Step 1: Identify Question Components】
    Break down the question into these elements:
    - TIME_DIMENSION: What time period? (e.g., 2015-2022, recent 3 years)
    - METRIC_TYPE: What to measure? (count, growth rate, average, ratio)
    - GROUPING: What to group by? (year, category, price range)
    - FILTERING: What conditions? (specific categories, thresholds)
    - COMPARISON: Compare with what? (overall average, year-over-year)
    - FINAL_OUTPUT: What does the user want to see?
    
    【Step 2: SQL Building Blocks】
    Map components to SQL patterns:
    
    A. TIME-BASED AGGREGATION:
       - Extract time period: CAST(SUBSTR(release_date, 1, 4) AS INTEGER) AS year
       - Filter range: WHERE release_date BETWEEN 'YYYY-01-01' AND 'YYYY-12-31'
       
    B. GROWTH RATE CALCULATION:
       - Year-over-year: LAG(metric) OVER (PARTITION BY category ORDER BY year)
       - Growth formula: (current - previous) * 1.0 / NULLIF(previous, 0)
       - CRITICAL: Always use NULLIF() or CASE to handle division by zero
       
    C. RATIO/RATE CALCULATION:
       - Review rate: positive * 1.0 / NULLIF(positive + negative, 0)
       - CRITICAL: Filter out records where denominator is 0 before calculation
       - Use CASE WHEN for NULL handling
       
    D. COMPARATIVE ANALYSIS:
       - Category-specific: Calculate metric for each group
       - Overall baseline: Calculate same metric for entire dataset
       - Comparison: Use CROSS JOIN to combine both
       
    E. MULTI-CRITERIA FILTERING:
       - Apply specific filters EARLY in the query (in first CTE)
       - Don't calculate first then filter - inefficient and may cause logic errors
    
    【Step 3: Common Pitfalls to Avoid】
    
    PITFALL #1: Division by Zero
    ❌ BAD:  value / denominator
    ✅ GOOD: value * 1.0 / NULLIF(denominator, 0)
    ✅ GOOD: CASE WHEN denominator > 0 THEN value * 1.0 / denominator ELSE NULL END
    
    PITFALL #2: Double Counting in JOINs
    ❌ BAD:  COUNT(*) when joining one-to-many
    ✅ GOOD: COUNT(DISTINCT primary_key)
    
    PITFALL #3: Incorrect Filter Placement
    ❌ BAD:  Calculate all → Filter specific items at the end
    ✅ GOOD: Filter specific items early → Calculate only what's needed
    
    PITFALL #4: LAG() Default Value
    ❌ BAD:  LAG(value, 1, 0) -- Can cause division by zero
    ✅ GOOD: LAG(value, 1) -- Returns NULL for first row
    
    PITFALL #5: Aggregating Before JOIN
    ❌ BAD:  JOIN then aggregate entire result
    ✅ GOOD: Aggregate in CTE, then JOIN aggregated results
    """

    # doc6="""
    # IMPORTANT: Querying categories and tags requires JOIN operations:
    
    # - The categories table is SEPARATE from the games table
    # - The tags table is SEPARATE from the games table
    # - To query game categories, you MUST use JOIN with the categories table
    # - To query game tags, you MUST use JOIN with the tags table
    
    # JOIN Examples:
    
    # 1. To find games by category (e.g., 'Indie'):
    #    SELECT g.name, g.max_owners 
    #    FROM games g
    #    JOIN categories c ON g.app_id = c.app_id
    #    WHERE c.categories = 'Indie'
    #    ORDER BY g.max_owners DESC LIMIT 10
    
    # 2. To find games with specific tags:
    #    SELECT g.name, g.price
    #    FROM games g
    #    JOIN tags t ON g.app_id = t.app_id
    #    WHERE t.tags LIKE '%Action%'
    #    ORDER BY g.positive DESC LIMIT 10
    
    # 3. To count games by category:
    #    SELECT c.categories, COUNT(*) as game_count
    #    FROM games g
    #    JOIN categories c ON g.app_id = c.app_id
    #    GROUP BY c.categories
    #    ORDER BY game_count DESC
    
    # 4. To get games with both category and tag filters:
    #    SELECT g.name, g.price, c.categories, t.tags
    #    FROM games g
    #    JOIN categories c ON g.app_id = c.app_id
    #    JOIN tags t ON g.app_id = t.app_id
    #    WHERE c.categories = 'RPG' AND t.tags LIKE '%Fantasy%'
    #    ORDER BY g.positive DESC LIMIT 10
    
    # Remember: NEVER query categories or tags directly from the games table!
    # """
    Documents=[doc1,doc2,doc3,doc4,doc5,doc7,doc8,doc9]
    sql_examples=[]
    sql_examples = [
   { "question":"""Find Multi-player games with >100,000 owners and >90% positive review rate. 
    Group by price range (free, $0-10, $10-30, $30+) and calculate average metrics per range.
    """,
    "sql":"""WITH qualified_games AS (
    -- STEP 1: Filter early with all criteria
    SELECT 
        g.*,
        CASE 
            WHEN (g.positive + g.negative) > 0 
            THEN g.positive * 1.0 / (g.positive + g.negative)
            ELSE 0
        END AS positive_rate
    FROM games g
    JOIN categories c ON g.app_id = c.app_id
    WHERE c.categories = 'Multi-player'
      AND g.max_owners > 100000
      AND (g.positive + g.negative) > 0  -- Ensure we can calculate rate
),
qualified_with_rate AS (
    -- STEP 2: Apply rate filter
    SELECT *
    FROM qualified_games
    WHERE positive_rate > 0.90
),
price_categorized AS (
    -- STEP 3: Categorize into price ranges
    SELECT 
        *,
        CASE 
            WHEN price = 0 THEN 'Free'
            WHEN price > 0 AND price <= 10 THEN 'Low ($0-10)'
            WHEN price > 10 AND price <= 30 THEN 'Mid ($10-30)'
            WHEN price > 30 THEN 'High ($30+)'
        END AS price_range
    FROM qualified_with_rate
)
-- STEP 4: Aggregate by price range
SELECT 
    price_range,
    COUNT(*) AS game_count,
    ROUND(AVG(price), 2) AS avg_price,
    ROUND(AVG((min_owners + max_owners) / 2.0), 0) AS avg_owners,
    ROUND(AVG(positive_rate) * 100, 2) || '%' AS avg_positive_rate
FROM price_categorized
GROUP BY price_range
ORDER BY 
    CASE price_range
        WHEN 'Free' THEN 1
        WHEN 'Low ($0-10)' THEN 2
        WHEN 'Mid ($10-30)' THEN 3
        WHEN 'High ($30+)' THEN 4
    END;
""",
    }
    ]
    #     {
    #         "question": "Which Indie game has the most owners?",
    #         "sql": "SELECT g.name FROM games g JOIN categories c ON g.app_id = c.app_id WHERE c.categories = 'Indie' ORDER BY g.max_owners DESC LIMIT 1"
    #     },
    #     {
    #         "question": "What are the top 5 Action games by positive reviews?",
    #         "sql": "SELECT g.name, g.positive FROM games g JOIN tags t ON g.app_id = t.app_id WHERE t.tags LIKE '%Action%' ORDER BY g.positive DESC LIMIT 5"
    #     },
    #     {
    #         "question": "How many games are in each category?",
    #         "sql": "SELECT c.categories, COUNT(*) as count FROM games g JOIN categories c ON g.app_id = c.app_id GROUP BY c.categories ORDER BY count DESC"
    #     }
    # ]
    for doc in Documents:
        VannaDB.train(documentation=doc)
    if sql_examples:
        for sql_example in sql_examples:
            VannaDB.train(question=sql_example["question"], sql=sql_example["sql"])
    print("VannaDB trained successfully")
    return VannaDB

if __name__ == "__main__":
    VannaDB=Setup_Vanna_VectorDB()
    Connect_to_SQLite(VannaDB)
    Train_VannaDB_On_SQLite(VannaDB)
    print("VannaDB trained successfully")

    