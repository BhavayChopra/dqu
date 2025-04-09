import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Database Configuration
def create_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            database="axe_assistant",  # Replace with actual DB name
            user="root",
            password="Bjrock_007",      # Replace with actual password
            port=3306,
            # Add buffer configuration to handle sort memory issues
            buffered=True,
            raise_on_warnings=True,
            # Add connection parameters to handle sort memory issues
            connection_timeout=300,
            use_pure=True,
            # Add parameters to handle sort memory issues
            consume_results=True,
            pool_size=5,
            pool_reset_session=True
        )
        
        # Set session variables to increase sort buffer size and other memory settings
        cursor = connection.cursor()
        cursor.execute("SET SESSION sort_buffer_size = 8388608")  # 8MB (increased from 4MB)
        cursor.execute("SET SESSION sql_buffer_result = ON")
        cursor.execute("SET SESSION join_buffer_size = 4194304")  # 4MB
        cursor.execute("SET SESSION tmp_table_size = 67108864")  # 64MB
        cursor.execute("SET SESSION max_heap_table_size = 67108864")  # 64MB
        cursor.close()
        return connection
    except Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

@st.cache_data  # Cache data without time limitation
def load_data(client_name=None, client_type=None, client_sector=None, client_country=None, user_id=None, date_range=None):
    connection = create_db_connection()
    if connection:
        try:
            # Build query with filters and deduplication
            # Use more efficient query with limited columns and better filtering
            query = """
                SELECT 
                    thread_id,
                    user_id,
                    client_name,
                    client_type,
                    client_sector,
                    client_country,
                    thread_created_on,
                    LEFT(user_prompt, 500) as user_prompt,
                    LEFT(response, 500) as response,
                    reaction,
                    feedback,
                    feedback_updated_on
                FROM unique_prompts_responses
                WHERE 1=1
            """
            
            params = []
            
            # Add filters to query
            if client_name:
                query += " AND client_name = %s"
                params.append(client_name)
                
            if client_type:
                query += " AND client_type = %s"
                params.append(client_type)
                
            if client_sector:
                query += " AND client_sector = %s"
                params.append(client_sector)
                
            if client_country:
                query += " AND client_country = %s"
                params.append(client_country)
                
            if user_id:
                query += " AND user_id = %s"
                params.append(user_id)
                
            if date_range:
                query += " AND thread_created_on BETWEEN %s AND %s"
                # Convert Python datetime objects to MySQL-compatible string format
                formatted_dates = [date.strftime('%Y-%m-%d %H:%M:%S') for date in date_range]
                params.extend(formatted_dates)
            else:
                # Default to last 3 months if no date range specified
                query += " AND thread_created_on >= DATE_SUB(NOW(), INTERVAL 3 MONTH)"
            
            # Order results by creation date
            query += " ORDER BY thread_created_on DESC"
            
            # Use a cursor with server-side cursor to reduce memory usage
            cursor = connection.cursor(dictionary=True, buffered=True)
            
            try:
                # Debug information for troubleshooting
                try:
                    cursor.execute(query, params)
                except Error as e:
                    st.error(f"Error executing query: {e}")
                    # Try to provide more specific error information for timestamp issues
                    if "timestamp" in str(e).lower():
                        st.error("There was an issue with date format conversion. Please try a different date range.")
                    return pd.DataFrame()
                
                # Fetch data in smaller chunks to avoid memory issues
                chunk_size = 200
                rows = []
                while True:
                    chunk = cursor.fetchmany(chunk_size)
                    if not chunk:
                        break
                    rows.extend(chunk)
                    
                df = pd.DataFrame(rows)
            except Error as e:
                st.error(f"Error executing query: {e}")
                return pd.DataFrame()
            finally:
                # Close cursor and connection
                cursor.close()
                connection.close()
            
            if df.empty:
                st.info("No data found with the current filters. Try adjusting your filter criteria or check database connection.")
                return pd.DataFrame()
            
            # Data processing with error handling
            try:
                # Convert date columns
                if 'thread_created_on' in df.columns:
                    df['thread_created_on'] = pd.to_datetime(df['thread_created_on'], errors='coerce')
                if 'feedback_updated_on' in df.columns:
                    df['feedback_updated_on'] = pd.to_datetime(df['feedback_updated_on'], errors='coerce')
                
                # Extract response time data
                def extract_response_time(response_text):
                    try:
                        if pd.isna(response_text) or response_text is None or response_text == '':
                            return None
                        
                        response_text = str(response_text)
                        # Look for the pattern <!--(Time taken for first response: X seconds) --->
                        import re
                        time_pattern = re.compile(r'<!--\(Time taken for first response: (\d+) seconds\) --->')
                        match = time_pattern.search(response_text)
                        
                        if match:
                            return int(match.group(1))
                        return None
                    except Exception as e:
                        return None
                
                if 'response' in df.columns:
                    df['response_time_seconds'] = df['response'].apply(extract_response_time)
                    # Convert to numeric to ensure proper analysis
                    df['response_time_seconds'] = pd.to_numeric(df['response_time_seconds'], errors='coerce')
                
                # Sentiment analysis with error handling
                def safe_sentiment(text):
                    try:
                        if pd.isna(text) or text is None or text == '':
                            return 0
                        return TextBlob(str(text)).sentiment.polarity
                    except Exception as e:
                        st.warning(f"Error in sentiment analysis: {e}")
                        return 0
                
                if 'user_prompt' in df.columns:
                    df['prompt_sentiment'] = df['user_prompt'].apply(safe_sentiment)
                if 'response' in df.columns:
                    df['response_sentiment'] = df['response'].apply(safe_sentiment)
                
                # Add sentiment explorer to help debug sentiment analysis
                st.sidebar.subheader("Sentiment Analysis Debug")
                show_sentiment_debug = st.sidebar.checkbox("Show Sentiment Analysis Debug")
                if show_sentiment_debug:
                    st.sidebar.write("Sentiment Distribution:")
                    if 'prompt_sentiment' in df.columns:
                        prompt_neg = (df['prompt_sentiment'] < -0.1).sum()
                        prompt_neutral = ((df['prompt_sentiment'] >= -0.1) & (df['prompt_sentiment'] <= 0.1)).sum()
                        prompt_pos = (df['prompt_sentiment'] > 0.1).sum()
                        st.sidebar.write(f"Prompt Sentiment: {prompt_neg} negative, {prompt_neutral} neutral, {prompt_pos} positive")
                    
                    if 'response_sentiment' in df.columns:
                        response_neg = (df['response_sentiment'] < -0.1).sum()
                        response_neutral = ((df['response_sentiment'] >= -0.1) & (df['response_sentiment'] <= 0.1)).sum()
                        response_pos = (df['response_sentiment'] > 0.1).sum()
                        st.sidebar.write(f"Response Sentiment: {response_neg} negative, {response_neutral} neutral, {response_pos} positive")
                
                # Categorize prompts by topic
                if 'user_prompt' in df.columns:
                    df['topic'] = df['user_prompt'].apply(categorize_prompt_topic)
                
                # Analyze correlation between response time and satisfaction
                def analyze_satisfaction_by_response_time(df):
                    try:
                        if 'response_time_seconds' not in df.columns or 'reaction' not in df.columns:
                            return None
                        
                        # Create a satisfaction metric based on reaction
                        df['satisfaction_score'] = df['reaction'].apply(
                            lambda x: 1 if x == 'thumbs-up' else (-1 if x == 'thumbs-down' else 0)
                        )
                        
                        # Calculate correlation between response time and satisfaction
                        correlation = df[['response_time_seconds', 'satisfaction_score']].corr().iloc[0, 1]
                        
                        # Create response time bins for analysis
                        df['response_time_bin'] = pd.cut(
                            df['response_time_seconds'], 
                            bins=[0, 2, 5, 10, 20, float('inf')],
                            labels=['0-2s', '2-5s', '5-10s', '10-20s', '20s+']
                        )
                        
                        # Calculate satisfaction rate by response time bin
                        satisfaction_by_time = df.groupby('response_time_bin').agg(
                            positive_rate=('satisfaction_score', lambda x: sum(x > 0) / len(x) if len(x) > 0 else 0),
                            negative_rate=('satisfaction_score', lambda x: sum(x < 0) / len(x) if len(x) > 0 else 0),
                            neutral_rate=('satisfaction_score', lambda x: sum(x == 0) / len(x) if len(x) > 0 else 0),
                            count=('satisfaction_score', 'count')
                        ).reset_index()
                        
                        return {
                            'correlation': correlation,
                            'satisfaction_by_time': satisfaction_by_time
                        }
                    except Exception as e:
                        return None
                
                # Additional insights and metrics
                def generate_additional_insights(df):
                    insights = {}
                    
                    try:
                        # 1. Response time distribution statistics
                        if 'response_time_seconds' in df.columns:
                            response_time_stats = {
                                'mean': df['response_time_seconds'].mean(),
                                'median': df['response_time_seconds'].median(),
                                'min': df['response_time_seconds'].min(),
                                'max': df['response_time_seconds'].max(),
                                'p90': df['response_time_seconds'].quantile(0.9),  # 90th percentile
                                'p95': df['response_time_seconds'].quantile(0.95)  # 95th percentile
                            }
                            insights['response_time_stats'] = response_time_stats
                        
                        # 2. Topic-specific response time analysis
                        if 'topic' in df.columns and 'response_time_seconds' in df.columns:
                            topic_response_times = df.groupby('topic').agg(
                                avg_response_time=('response_time_seconds', 'mean'),
                                median_response_time=('response_time_seconds', 'median'),
                                count=('response_time_seconds', 'count')
                            ).reset_index()
                            insights['topic_response_times'] = topic_response_times
                        
                        # 3. Client-specific satisfaction analysis
                        if 'client_name' in df.columns and 'satisfaction_score' in df.columns:
                            client_satisfaction = df.groupby('client_name').agg(
                                avg_satisfaction=('satisfaction_score', 'mean'),
                                positive_rate=('satisfaction_score', lambda x: sum(x > 0) / len(x) if len(x) > 0 else 0),
                                negative_rate=('satisfaction_score', lambda x: sum(x < 0) / len(x) if len(x) > 0 else 0),
                                avg_response_time=('response_time_seconds', 'mean'),
                                count=('satisfaction_score', 'count')
                            ).reset_index()
                            insights['client_satisfaction'] = client_satisfaction
                        
                        # 4. Trend analysis of response times over time
                        if 'thread_created_on' in df.columns and 'response_time_seconds' in df.columns:
                            # Group by week
                            time_trend = df.groupby(pd.Grouper(key='thread_created_on', freq='W')).agg(
                                avg_response_time=('response_time_seconds', 'mean'),
                                median_response_time=('response_time_seconds', 'median'),
                                avg_satisfaction=('satisfaction_score', 'mean'),
                                count=('response_time_seconds', 'count')
                            ).reset_index()
                            insights['time_trend'] = time_trend
                        
                        # 5. Response time threshold analysis
                        # Find the optimal response time threshold for satisfaction
                        if 'response_time_seconds' in df.columns and 'satisfaction_score' in df.columns:
                            thresholds = range(1, 21)  # Test thresholds from 1 to 20 seconds
                            threshold_results = []
                            
                            for threshold in thresholds:
                                fast_responses = df[df['response_time_seconds'] <= threshold]
                                slow_responses = df[df['response_time_seconds'] > threshold]
                                
                                if len(fast_responses) > 0 and len(slow_responses) > 0:
                                    fast_satisfaction = fast_responses['satisfaction_score'].mean()
                                    slow_satisfaction = slow_responses['satisfaction_score'].mean()
                                    satisfaction_diff = fast_satisfaction - slow_satisfaction
                                    
                                    threshold_results.append({
                                        'threshold': threshold,
                                        'fast_satisfaction': fast_satisfaction,
                                        'slow_satisfaction': slow_satisfaction,
                                        'satisfaction_diff': satisfaction_diff,
                                        'fast_count': len(fast_responses),
                                        'slow_count': len(slow_responses)
                                    })
                            
                            if threshold_results:
                                threshold_df = pd.DataFrame(threshold_results)
                                # Find optimal threshold (maximum satisfaction difference)
                                optimal_threshold = threshold_df.loc[threshold_df['satisfaction_diff'].idxmax()]
                                insights['optimal_response_threshold'] = optimal_threshold
                                insights['threshold_analysis'] = threshold_df
                        
                        # 6. NEW: Response time by reaction type analysis
                        if 'reaction' in df.columns and 'response_time_seconds' in df.columns:
                            # Filter to only include actual reactions (thumbs-up/thumbs-down)
                            reaction_df = df[df['reaction'].isin(['thumbs-up', 'thumbs-down'])]
                            if not reaction_df.empty:
                                reaction_response_times = reaction_df.groupby('reaction').agg(
                                    avg_response_time=('response_time_seconds', 'mean'),
                                    median_response_time=('response_time_seconds', 'median'),
                                    min_response_time=('response_time_seconds', 'min'),
                                    max_response_time=('response_time_seconds', 'max'),
                                    count=('response_time_seconds', 'count')
                                ).reset_index()
                                insights['reaction_response_times'] = reaction_response_times
                        
                        # 7. NEW: Response time by sentiment analysis
                        if 'response_time_seconds' in df.columns and 'prompt_sentiment' in df.columns and 'response_sentiment' in df.columns:
                            # Create sentiment bins
                            df['prompt_sentiment_bin'] = pd.cut(
                                df['prompt_sentiment'], 
                                bins=[-1.0, -0.5, 0, 0.5, 1.0],
                                labels=['Very Negative', 'Negative', 'Positive', 'Very Positive']
                            )
                            df['response_sentiment_bin'] = pd.cut(
                                df['response_sentiment'], 
                                bins=[-1.0, -0.5, 0, 0.5, 1.0],
                                labels=['Very Negative', 'Negative', 'Positive', 'Very Positive']
                            )
                            
                            # Analyze response time by prompt sentiment
                            prompt_sentiment_response_times = df.groupby('prompt_sentiment_bin').agg(
                                avg_response_time=('response_time_seconds', 'mean'),
                                median_response_time=('response_time_seconds', 'median'),
                                count=('response_time_seconds', 'count')
                            ).reset_index()
                            insights['prompt_sentiment_response_times'] = prompt_sentiment_response_times
                            
                            # Analyze response time by response sentiment
                            response_sentiment_response_times = df.groupby('response_sentiment_bin').agg(
                                avg_response_time=('response_time_seconds', 'mean'),
                                median_response_time=('response_time_seconds', 'median'),
                                count=('response_time_seconds', 'count')
                            ).reset_index()
                            insights['response_sentiment_response_times'] = response_sentiment_response_times
                        
                        # 8. NEW: Response time by client type and sector
                        if 'response_time_seconds' in df.columns:
                            if 'client_type' in df.columns:
                                client_type_response_times = df.groupby('client_type').agg(
                                    avg_response_time=('response_time_seconds', 'mean'),
                                    median_response_time=('response_time_seconds', 'median'),
                                    count=('response_time_seconds', 'count')
                                ).reset_index()
                                insights['client_type_response_times'] = client_type_response_times
                            
                            if 'client_sector' in df.columns:
                                client_sector_response_times = df.groupby('client_sector').agg(
                                    avg_response_time=('response_time_seconds', 'mean'),
                                    median_response_time=('response_time_seconds', 'median'),
                                    count=('response_time_seconds', 'count')
                                ).reset_index()
                                insights['client_sector_response_times'] = client_sector_response_times
                        
                        return insights
                    except Exception as e:
                        return {'error': str(e)}
                
                # Store correlation analysis results
                df.correlation_analysis = analyze_satisfaction_by_response_time(df)
                
                # Store additional insights
                df.additional_insights = generate_additional_insights(df)
                
                # Remove duplicates after processing
                if 'thread_id' in df.columns and 'user_prompt' in df.columns and 'response' in df.columns:
                    df = df.drop_duplicates(subset=['thread_id', 'user_prompt', 'response'])
                
                return df
            except Exception as e:
                st.error(f"Error processing data: {e}")
                return pd.DataFrame()
        except Error as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# Function to categorize prompts by topic using ML/NLP approach
def categorize_prompt_topic(prompt, n_topics=10):
    try:
        # Handle empty or invalid prompts
        if pd.isna(prompt) or prompt is None or prompt == "":
            return "unknown"
            
        prompt = str(prompt).lower()
        
        # Define topic names that will be assigned to clusters
        # These are the same categories as before for consistency
        topic_names = [
            'forms', 'tables', 'images', 'navigation', 'color',
            'keyboard', 'screen reader', 'mobile', 'pdf', 'wcag'
        ]
        
        # Use a singleton pattern to avoid retraining the model for each prompt
        if not hasattr(categorize_prompt_topic, "vectorizer"):
            # Initialize and train the model only once
            # This is a static variable that persists between function calls
            categorize_prompt_topic.vectorizer = CountVectorizer(
                max_df=0.95, min_df=2, stop_words='english', max_features=1000
            )
            categorize_prompt_topic.lda = LatentDirichletAllocation(
                n_components=n_topics, random_state=42, max_iter=10
            )
            categorize_prompt_topic.trained = False
            categorize_prompt_topic.all_prompts = []
        
        # Add current prompt to collection for batch processing
        categorize_prompt_topic.all_prompts.append(prompt)
        
        # If we have enough prompts or if this is a subsequent call after training
        if len(categorize_prompt_topic.all_prompts) >= 50 or categorize_prompt_topic.trained:
            if not categorize_prompt_topic.trained:
                # Train the model on collected prompts
                try:
                    # Create document-term matrix
                    dtm = categorize_prompt_topic.vectorizer.fit_transform(categorize_prompt_topic.all_prompts)
                    
                    # Train LDA model
                    categorize_prompt_topic.lda.fit(dtm)
                    categorize_prompt_topic.trained = True
                    
                    # Get feature names for interpretation
                    feature_names = categorize_prompt_topic.vectorizer.get_feature_names_out()
                    
                    # Store top words for each topic for interpretation
                    categorize_prompt_topic.topic_keywords = []
                    for topic_idx, topic in enumerate(categorize_prompt_topic.lda.components_):
                        top_keywords_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 words
                        top_keywords = [feature_names[i] for i in top_keywords_idx]
                        categorize_prompt_topic.topic_keywords.append(top_keywords)
                except Exception as e:
                    # Fallback to simple classification if training fails
                    return "other"
            
            # Transform the current prompt
            prompt_vector = categorize_prompt_topic.vectorizer.transform([prompt])
            
            # Get topic distribution for the prompt
            topic_distribution = categorize_prompt_topic.lda.transform(prompt_vector)[0]
            
            # Get the dominant topic
            dominant_topic_idx = topic_distribution.argmax()
            
            # Return the corresponding topic name
            return topic_names[dominant_topic_idx]
        else:
            # Not enough data to train yet, use a simple fallback
            return "pending_classification"
    except Exception as e:
        # Return a default value if any error occurs
        return "other"

# Function to extract top keywords for each topic (for visualization)
def get_topic_keywords():
    if hasattr(categorize_prompt_topic, "topic_keywords"):
        topic_names = [
            'forms', 'tables', 'images', 'navigation', 'color',
            'keyboard', 'screen reader', 'mobile', 'pdf', 'wcag'
        ]
        return {topic: keywords for topic, keywords in zip(topic_names, categorize_prompt_topic.topic_keywords)}
    return {}

# Main App
def main():
    st.set_page_config(page_title="Client Interaction Analytics", layout="wide")
    st.title("Client Interaction Analysis Dashboard")
    
    # Initialize session state for filters if not exists
    if 'client_filter' not in st.session_state:
        st.session_state.client_filter = None
    if 'user_filter' not in st.session_state:
        st.session_state.user_filter = None
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Get initial data to populate filter options
    try:
        initial_df = load_data()
        
        if initial_df.empty:
            st.warning("No data available from database. Please check your connection settings.")
            st.info("This dashboard requires a MySQL database with client interaction data.")
            return
        
        # Client filters
        st.sidebar.subheader("Client Filters")
        client_names = ["All"] + sorted(initial_df['client_name'].dropna().unique().tolist())
        selected_client = st.sidebar.selectbox("Client Name", client_names)
        
        client_types = ["All"] + sorted(initial_df['client_type'].dropna().unique().tolist())
        selected_type = st.sidebar.selectbox("Client Type", client_types)
        
        client_sectors = ["All"] + sorted(initial_df['client_sector'].dropna().unique().tolist())
        selected_sector = st.sidebar.selectbox("Client Sector", client_sectors)
        
        client_countries = ["All"] + sorted(initial_df['client_country'].dropna().unique().tolist())
        selected_country = st.sidebar.selectbox("Client Country", client_countries)
        
        # User filter
        st.sidebar.subheader("User Filter")
        user_ids = ["All"] + sorted(initial_df['user_id'].dropna().unique().tolist())
        selected_user = st.sidebar.selectbox("User ID", user_ids)
        
        # Date range filter
        st.sidebar.subheader("Date Range")
        if 'thread_created_on' in initial_df.columns and not initial_df['thread_created_on'].empty:
            date_col = 'thread_created_on'
            min_date = initial_df[date_col].min().date()
            max_date = initial_df[date_col].max().date()
            date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
        else:
            st.sidebar.info("Date information not available")
            date_range = None
        
        # Apply filters to load data
        client_name_param = selected_client if selected_client != "All" else None
        client_type_param = selected_type if selected_type != "All" else None
        client_sector_param = selected_sector if selected_sector != "All" else None
        client_country_param = selected_country if selected_country != "All" else None
        user_id_param = selected_user if selected_user != "All" else None
        # Create date range parameter with proper datetime objects
        date_range_param = None
        if date_range and len(date_range) == 2:
            try:
                # Convert to datetime objects first to ensure proper format
                date_range_param = [pd.Timestamp(date_range[0]).to_pydatetime(), 
                                   pd.Timestamp(date_range[1]).replace(hour=23, minute=59, second=59).to_pydatetime()]
            except Exception as e:
                st.error(f"Error formatting date range: {e}")
                date_range_param = None
        
        # Load filtered data
        with st.spinner("Loading data..."):
            df = load_data(
                client_name=client_name_param,
                client_type=client_type_param,
                client_sector=client_sector_param,
                client_country=client_country_param,
                user_id=user_id_param,
                date_range=date_range_param
            )
        
        if df.empty:
            st.warning("No data matches the selected filters. Please adjust your filter criteria.")
            st.info("Try selecting a broader date range, removing some filters, or checking if the database contains data for the selected parameters.")
            return
        
        # Add a utility function for data exploration
        def create_data_explorer(data, title, columns_to_show=None):
            st.subheader(title)
            
            # Default columns to show if not specified
            if columns_to_show is None:
                columns_to_show = ['thread_id', 'user_prompt', 'response', 'reaction', 'response_time_seconds', 'topic']
            
            # Ensure all requested columns exist in the dataframe
            valid_columns = [col for col in columns_to_show if col in data.columns]
            
            # Display data count
            st.write(f"Total records: {len(data)}")
            
            # Add search functionality
            search_term = st.text_input(f"Search in {title}", "")
            
            # Filter data based on search term if provided
            if search_term:
                filtered_data = data[data.apply(lambda row: any(search_term.lower() in str(cell).lower() 
                                                            for cell in row if isinstance(cell, (str, int, float))), axis=1)]
                st.write(f"Found {len(filtered_data)} records matching '{search_term}'")
            else:
                filtered_data = data
            
            # Add pagination
            page_size = st.slider(f"Records per page for {title}", min_value=5, max_value=50, value=10, step=5)
            total_pages = max(1, (len(filtered_data) + page_size - 1) // page_size)
            
            if total_pages > 1:
                page_number = st.number_input(f"Page (1-{total_pages}) for {title}", min_value=1, max_value=total_pages, value=1)
            else:
                page_number = 1
            
            # Calculate start and end indices for the current page
            start_idx = (page_number - 1) * page_size
            end_idx = min(start_idx + page_size, len(filtered_data))
            
            # Display the paginated data
            st.dataframe(filtered_data.iloc[start_idx:end_idx][valid_columns], use_container_width=True)
            
            # Add download button for the full dataset
            csv = filtered_data[valid_columns].to_csv(index=False)
            st.download_button(
                label=f"Download {title} data as CSV",
                data=csv,
                file_name=f"{title.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
            return filtered_data
        
        # KPI Metrics
        st.header("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Interactions", len(df))
        with col2:
            if 'reaction' in df.columns:
                positive_count = df[df['reaction'] == 'thumbs-up'].shape[0]
                negative_count = df[df['reaction'] == 'thumbs-down'].shape[0]
                total_reactions = positive_count + negative_count
                positive_rate = positive_count / total_reactions if total_reactions > 0 else 0
                st.metric("Positive Reaction Rate", f"{positive_rate:.1%}")
            else:
                st.metric("Positive Reaction Rate", "N/A")
        with col3:
            if 'response_time_seconds' in df.columns:
                avg_response_time = df['response_time_seconds'].mean()
                st.metric("Avg Response Time", f"{avg_response_time:.1f}s")
            else:
                st.metric("Avg Response Time", "N/A")
        with col4:
            if 'feedback' in df.columns:
                feedback_count = df['feedback'].notna().sum()
                st.metric("Feedback Responses", feedback_count)
            else:
                st.metric("Feedback Responses", "N/A")
        
        # Main Visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Trend Analysis", "Content Analysis", "Topic Analysis", "Response Time Analysis", "Client Insights"])
        
        with tab1:
            st.subheader("Reaction Trends Over Time")
            
            # Create a time series dataframe with proper handling of empty data
            if not df.empty and 'reaction' in df.columns and 'thread_created_on' in df.columns:
                # Create a copy of reaction column with standardized values
                df['reaction_std'] = df['reaction'].fillna('none')
                
                # Group by week and reaction
                try:
                    time_data = df.groupby([pd.Grouper(key='thread_created_on', freq='W'), 'reaction_std']).size().reset_index()
                    time_data.columns = ['thread_created_on', 'reaction', 'count']
                    
                    # Create the time series plot
                    fig = px.line(time_data, x='thread_created_on', y='count', color='reaction',
                                title="Weekly Interaction Trends",
                                labels={'count': 'Number of Interactions', 'thread_created_on': 'Date'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add data explorer for reactions
                    with st.expander("Explore Reaction Data"):
                        reaction_type = st.selectbox(
                            "Select reaction type to explore",
                            ["thumbs-up", "thumbs-down", "none"]
                        )
                        reaction_data = df[df['reaction_std'] == reaction_type]
                        create_data_explorer(
                            reaction_data,
                            f"{reaction_type.capitalize()} Reactions",
                            ['thread_id', 'user_prompt', 'response', 'thread_created_on', 'response_time_seconds', 'topic']
                        )
                    
                    # Add response time trend if available
                    if hasattr(df, 'additional_insights') and 'time_trend' in df.additional_insights:
                        time_trend = df.additional_insights['time_trend']
                        
                        # Create response time trend plot
                        fig2 = px.line(time_trend, x='thread_created_on', 
                                      y=['avg_response_time', 'median_response_time'],
                                      title="Response Time Trends Over Time",
                                      labels={
                                          'thread_created_on': 'Date', 
                                          'value': 'Response Time (seconds)',
                                          'variable': 'Metric'
                                      })
                        fig2.update_layout(legend_title_text='Metric')
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Create satisfaction trend plot
                        if 'avg_satisfaction' in time_trend.columns:
                            fig3 = px.line(time_trend, x='thread_created_on', y='avg_satisfaction',
                                          title="Satisfaction Trend Over Time",
                                          labels={
                                              'thread_created_on': 'Date',
                                              'avg_satisfaction': 'Average Satisfaction Score'
                                          })
                            st.plotly_chart(fig3, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating time series plot: {e}")
                
                # Add sentiment data explorer
                with st.expander("Explore Sentiment Data"):
                    sentiment_type = st.radio(
                        "Select sentiment type to explore",
                        ["Prompt Sentiment", "Response Sentiment"]
                    )
                    
                    sentiment_column = 'prompt_sentiment' if sentiment_type == "Prompt Sentiment" else 'response_sentiment'
                    
                    sentiment_range = st.select_slider(
                        "Select sentiment range to explore",
                        options=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
                        value=("Negative", "Negative")
                    )
                    
                    # Map sentiment ranges to numeric values
                    sentiment_ranges = {
                        "Very Negative": (-1.0, -0.6),
                        "Negative": (-0.6, -0.2),
                        "Neutral": (-0.2, 0.2),
                        "Positive": (0.2, 0.6),
                        "Very Positive": (0.6, 1.0)
                    }
                    
                    start_range = sentiment_ranges[sentiment_range[0]][0]
                    end_range = sentiment_ranges[sentiment_range[1]][1]
                    
                    sentiment_data = df[(df[sentiment_column] >= start_range) & 
                                      (df[sentiment_column] <= end_range)]
                    
                    create_data_explorer(
                        sentiment_data,
                        f"{sentiment_range[0]} to {sentiment_range[1]} {sentiment_type}",
                        ['thread_id', 'user_prompt', 'response', sentiment_column, 'reaction', 'response_time_seconds']
                    )
                
                # Sentiment over time
                try:
                    if 'prompt_sentiment' in df.columns and 'response_sentiment' in df.columns:
                        sentiment_time = df.groupby(pd.Grouper(key='thread_created_on', freq='W')).agg({
                            'prompt_sentiment': 'mean',
                            'response_sentiment': 'mean'
                        }).reset_index()
                        
                        fig2 = px.line(sentiment_time, x='thread_created_on', 
                                    y=['prompt_sentiment', 'response_sentiment'],
                                    title="Sentiment Trends Over Time",
                                    labels={
                                        'value': 'Sentiment Score',
                                        'variable': 'Sentiment Type',
                                        'thread_created_on': 'Date'
                                    })
                        st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating sentiment time series: {e}")
            else:
                st.info("Not enough data to display time trends.")
            
        with tab2:
            st.subheader("Content Analysis")
            
            # Word Cloud for Prompts
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Common Themes in User Prompts")
                if not df.empty and 'user_prompt' in df.columns and df['user_prompt'].notna().sum() > 0:
                    try:
                        text = ' '.join(df['user_prompt'].dropna().astype(str))
                        if text.strip():
                            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
                            plt.figure(figsize=(10, 5))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis("off")
                            st.pyplot(plt)
                        else:
                            st.info("Not enough text data for word cloud.")
                    except Exception as e:
                        st.error(f"Error generating word cloud: {e}")
                else:
                    st.info("No prompt data available.")
                    
            with col2:
                st.write("Common Themes in Responses")
                if not df.empty and 'response' in df.columns and df['response'].notna().sum() > 0:
                    try:
                        text = ' '.join(df['response'].dropna().astype(str))
                        if text.strip():
                            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
                            plt.figure(figsize=(10, 5))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis("off")
                            st.pyplot(plt)
                        else:
                            st.info("Not enough text data for word cloud.")
                    except Exception as e:
                        st.error(f"Error generating word cloud: {e}")
                else:
                    st.info("No response data available.")
            
            # Feedback Analysis
            st.subheader("Feedback Analysis")
            if not df.empty and 'feedback' in df.columns and df['feedback'].notna().sum() > 0:
                feedback_df = df[df['feedback'].notna()]
                
                # Display sample feedback
                with st.expander("View Sample Feedback"):
                    st.dataframe(feedback_df[['user_prompt', 'feedback']].head(10))
                    
                # Sentiment of feedback
                if len(feedback_df) > 0:
                    try:
                        feedback_df['feedback_sentiment'] = feedback_df['feedback'].apply(
                            lambda x: TextBlob(str(x)).sentiment.polarity if not pd.isna(x) else 0)
                        
                        fig = px.histogram(feedback_df, x='feedback_sentiment', 
                                        title="Distribution of Feedback Sentiment",
                                        labels={'feedback_sentiment': 'Sentiment Score'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add data explorer for feedback sentiment
                        with st.expander("Explore Feedback Sentiment Data"):
                            sentiment_range = st.select_slider(
                                "Select sentiment range to explore",
                                options=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
                                value=("Negative", "Negative")
                            )
                            
                            # Map sentiment ranges to numeric values
                            sentiment_ranges = {
                                "Very Negative": (-1.0, -0.6),
                                "Negative": (-0.6, -0.2),
                                "Neutral": (-0.2, 0.2),
                                "Positive": (0.2, 0.6),
                                "Very Positive": (0.6, 1.0)
                            }
                            
                            start_range = sentiment_ranges[sentiment_range[0]][0]
                            end_range = sentiment_ranges[sentiment_range[1]][1]
                            
                            sentiment_data = feedback_df[(feedback_df['feedback_sentiment'] >= start_range) & 
                                                        (feedback_df['feedback_sentiment'] <= end_range)]
                            
                            create_data_explorer(
                                sentiment_data,
                                f"{sentiment_range[0]} to {sentiment_range[1]} Sentiment",
                                ['thread_id', 'user_prompt', 'response', 'feedback', 'feedback_sentiment', 'response_time_seconds']
                            )
                    except Exception as e:
                        st.error(f"Error analyzing feedback sentiment: {e}")
            else:
                st.info("No feedback data available for analysis.")
                
        with tab3:
            st.subheader("Topic Analysis")
            
            if 'topic' in df.columns and df['topic'].notna().sum() > 0:
                try:
                    # Topic distribution
                    topic_counts = df['topic'].value_counts().reset_index()
                    topic_counts.columns = ['Topic', 'Count']
                    
                    fig = px.pie(topic_counts, values='Count', names='Topic', 
                                title="Distribution of Topics in User Prompts")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating topic distribution chart: {e}")
                
                try:
                    # Topic sentiment analysis
                    if 'prompt_sentiment' in df.columns and 'response_sentiment' in df.columns:
                        topic_sentiment = df.groupby('topic').agg({
                            'prompt_sentiment': 'mean',
                            'response_sentiment': 'mean',
                            'thread_id': 'count'
                        }).reset_index()
                        topic_sentiment.columns = ['Topic', 'Prompt Sentiment', 'Response Sentiment', 'Count']
                        
                        fig = px.scatter(topic_sentiment, x='Prompt Sentiment', y='Response Sentiment',
                                        size='Count', color='Topic', hover_name='Topic',
                                        title="Topic Sentiment Analysis")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating topic sentiment analysis: {e}")
                
                try:
                    # Topic reaction analysis
                    if 'reaction' in df.columns and df['reaction'].notna().sum() > 0:
                        topic_reaction = pd.crosstab(df['topic'], df['reaction'])
                        topic_reaction = topic_reaction.reset_index()
                        topic_reaction_melted = pd.melt(topic_reaction, id_vars=['topic'], 
                                                    var_name='Reaction', value_name='Count')
                        
                        fig = px.bar(topic_reaction_melted, x='topic', y='Count', color='Reaction',
                                    title="Reactions by Topic",
                                    labels={'topic': 'Topic'})
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating topic reaction analysis: {e}")
                
                # Display topic keywords if available
                if 'get_topic_keywords' in globals():
                    topic_keywords = get_topic_keywords()
                    if topic_keywords:
                        st.subheader("Topic Keywords")
                        for topic, keywords in topic_keywords.items():
                            st.write(f"**{topic}**: {', '.join(keywords)}")
            else:
                st.info("Topic categorization not available or insufficient data.")
                
        with tab4:
            st.subheader("Response Time Analysis")
            
            # Display response time distribution
            if 'response_time_seconds' in df.columns:
                st.write("### Response Time Distribution")
                
                # Create histogram of response times
                fig = px.histogram(df, x='response_time_seconds', nbins=20,
                                  title="Distribution of Response Times",
                                  labels={'response_time_seconds': 'Response Time (seconds)'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Add data explorer for response times
                with st.expander("Explore Response Time Data"):
                    response_time_range = st.slider(
                        "Select response time range to explore (seconds)",
                        min_value=int(df['response_time_seconds'].min()) if not pd.isna(df['response_time_seconds'].min()) else 0,
                        max_value=int(df['response_time_seconds'].max()) if not pd.isna(df['response_time_seconds'].max()) else 30,
                        value=(0, 10)
                    )
                    
                    response_time_data = df[(df['response_time_seconds'] >= response_time_range[0]) & 
                                          (df['response_time_seconds'] <= response_time_range[1])]
                    
                    create_data_explorer(
                        response_time_data,
                        f"Response Time {response_time_range[0]}-{response_time_range[1]} seconds",
                        ['thread_id', 'user_prompt', 'response', 'reaction', 'response_time_seconds', 'topic']
                    )
                
                # Display response time statistics
                if hasattr(df, 'additional_insights') and 'response_time_stats' in df.additional_insights:
                    stats = df.additional_insights['response_time_stats']
                    st.write("### Response Time Statistics")
                    
                    # Create two columns for statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Response Time", f"{stats['mean']:.2f}s")
                        st.metric("Median Response Time", f"{stats['median']:.2f}s")
                        st.metric("Minimum Response Time", f"{stats['min']:.2f}s")
                    with col2:
                        st.metric("Maximum Response Time", f"{stats['max']:.2f}s")
                        st.metric("90th Percentile", f"{stats['p90']:.2f}s")
                        st.metric("95th Percentile", f"{stats['p95']:.2f}s")
                
                # Display correlation between response time and satisfaction
                if hasattr(df, 'correlation_analysis') and df.correlation_analysis:
                    st.write("### Correlation with User Satisfaction")
                    
                    # Display correlation coefficient
                    correlation = df.correlation_analysis.get('correlation')
                    if correlation is not None:
                        st.metric("Correlation Coefficient", f"{correlation:.3f}", 
                                 delta="-1 to 1 scale, negative means faster responses correlate with higher satisfaction" if correlation < 0 else "1 to 1 scale, positive means slower responses correlate with higher satisfaction")
                    
                    # Display satisfaction by response time bin
                    satisfaction_by_time = df.correlation_analysis.get('satisfaction_by_time')
                    if satisfaction_by_time is not None:
                        # Create bar chart for satisfaction by response time bin
                        fig = px.bar(satisfaction_by_time, x='response_time_bin', 
                                    y=['positive_rate', 'negative_rate', 'neutral_rate'],
                                    title="Satisfaction Rates by Response Time",
                                    labels={
                                        'response_time_bin': 'Response Time',
                                        'value': 'Rate',
                                        'variable': 'Reaction Type'
                                    },
                                    barmode='group')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display count by response time bin
                        fig2 = px.bar(satisfaction_by_time, x='response_time_bin', y='count',
                                     title="Number of Interactions by Response Time",
                                     labels={
                                         'response_time_bin': 'Response Time',
                                         'count': 'Number of Interactions'
                                     })
                        st.plotly_chart(fig2, use_container_width=True)
                
                # NEW: Display response time by reaction type
                if hasattr(df, 'additional_insights') and 'reaction_response_times' in df.additional_insights:
                    st.write("### Response Time by Reaction Type")
                    
                    reaction_times = df.additional_insights['reaction_response_times']
                    
                    # Create bar chart for reaction response times
                    fig = px.bar(reaction_times, x='reaction', y=['avg_response_time', 'median_response_time'],
                                title="Response Time by Reaction Type",
                                labels={
                                    'reaction': 'Reaction',
                                    'value': 'Response Time (seconds)',
                                    'variable': 'Metric'
                                },
                                barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add data explorer for reaction response times
                    with st.expander("Explore Reaction Response Time Data"):
                        selected_reaction = st.selectbox(
                            "Select reaction type",
                            ["thumbs-up", "thumbs-down"]
                        )
                        
                        reaction_data = df[df['reaction'] == selected_reaction]
                        create_data_explorer(
                            reaction_data,
                            f"{selected_reaction} Reaction Data",
                            ['thread_id', 'user_prompt', 'response', 'response_time_seconds', 'topic']
                        )
                
                # NEW: Display response time by sentiment
                if hasattr(df, 'additional_insights') and 'prompt_sentiment_response_times' in df.additional_insights:
                    st.write("### Response Time by Sentiment")
                    
                    prompt_sentiment_times = df.additional_insights['prompt_sentiment_response_times']
                    response_sentiment_times = df.additional_insights.get('response_sentiment_response_times')
                    
                    # Create tabs for prompt and response sentiment
                    sentiment_tab1, sentiment_tab2 = st.tabs(["Prompt Sentiment", "Response Sentiment"])
                    
                    with sentiment_tab1:
                        # Create bar chart for prompt sentiment response times
                        fig = px.bar(prompt_sentiment_times, x='prompt_sentiment_bin', 
                                    y=['avg_response_time', 'median_response_time'],
                                    title="Response Time by Prompt Sentiment",
                                    labels={
                                        'prompt_sentiment_bin': 'Prompt Sentiment',
                                        'value': 'Response Time (seconds)',
                                        'variable': 'Metric'
                                    },
                                    barmode='group')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create scatter plot for count vs response time
                        fig2 = px.scatter(prompt_sentiment_times, x='avg_response_time', y='count',
                                        size='count', color='prompt_sentiment_bin',
                                        title="Response Time vs. Frequency by Prompt Sentiment",
                                        labels={
                                            'avg_response_time': 'Average Response Time (seconds)',
                                            'count': 'Number of Interactions',
                                            'prompt_sentiment_bin': 'Prompt Sentiment'
                                        })
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    with sentiment_tab2:
                        if response_sentiment_times is not None:
                            # Create bar chart for response sentiment response times
                            fig = px.bar(response_sentiment_times, x='response_sentiment_bin', 
                                        y=['avg_response_time', 'median_response_time'],
                                        title="Response Time by Response Sentiment",
                                        labels={
                                            'response_sentiment_bin': 'Response Sentiment',
                                            'value': 'Response Time (seconds)',
                                            'variable': 'Metric'
                                        },
                                        barmode='group')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create scatter plot for count vs response time
                            fig2 = px.scatter(response_sentiment_times, x='avg_response_time', y='count',
                                            size='count', color='response_sentiment_bin',
                                            title="Response Time vs. Frequency by Response Sentiment",
                                            labels={
                                                'avg_response_time': 'Average Response Time (seconds)',
                                                'count': 'Number of Interactions',
                                                'response_sentiment_bin': 'Response Sentiment'
                                            })
                            st.plotly_chart(fig2, use_container_width=True)
                
                # Display optimal response time threshold analysis
                if hasattr(df, 'additional_insights') and 'optimal_response_threshold' in df.additional_insights:
                    st.write("### Optimal Response Time Threshold")
                    
                    optimal = df.additional_insights['optimal_response_threshold']
                    threshold_analysis = df.additional_insights.get('threshold_analysis')
                    
                    # Display optimal threshold
                    st.metric("Optimal Response Time Threshold", f"{optimal['threshold']:.0f} seconds",
                             delta=f"Satisfaction difference: {optimal['satisfaction_diff']:.3f}")
                    
                    # Create line chart for threshold analysis
                    if threshold_analysis is not None:
                        fig = px.line(threshold_analysis, x='threshold', 
                                     y=['fast_satisfaction', 'slow_satisfaction', 'satisfaction_diff'],
                                     title="Satisfaction by Response Time Threshold",
                                     labels={
                                         'threshold': 'Response Time Threshold (seconds)',
                                         'value': 'Satisfaction Score',
                                         'variable': 'Metric'
                                     })
                        st.plotly_chart(fig, use_container_width=True)
                
                # NEW: Display response time by client type and sector
                if hasattr(df, 'additional_insights'):
                    # Client type response times
                    if 'client_type_response_times' in df.additional_insights:
                        st.write("### Response Time by Client Type")
                        
                        client_type_times = df.additional_insights['client_type_response_times']
                        
                        # Create bar chart for client type response times
                        fig = px.bar(client_type_times, x='client_type', y=['avg_response_time', 'median_response_time'],
                                    title="Response Time by Client Type",
                                    labels={
                                        'client_type': 'Client Type',
                                        'value': 'Response Time (seconds)',
                                        'variable': 'Metric'
                                    },
                                    barmode='group')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Client sector response times
                    if 'client_sector_response_times' in df.additional_insights:
                        st.write("### Response Time by Client Sector")
                        
                        client_sector_times = df.additional_insights['client_sector_response_times']
                        
                        # Create bar chart for client sector response times
                        fig = px.bar(client_sector_times, x='client_sector', y=['avg_response_time', 'median_response_time'],
                                    title="Response Time by Client Sector",
                                    labels={
                                        'client_sector': 'Client Sector',
                                        'value': 'Response Time (seconds)',
                                        'variable': 'Metric'
                                    },
                                    barmode='group')
                        fig.update_layout(xaxis={'categoryorder': 'total descending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create scatter plot for count vs response time by sector
                        fig2 = px.scatter(client_sector_times, x='avg_response_time', y='count',
                                        size='count', color='client_sector',
                                        title="Response Time vs. Frequency by Client Sector",
                                        labels={
                                            'avg_response_time': 'Average Response Time (seconds)',
                                            'count': 'Number of Interactions',
                                            'client_sector': 'Client Sector'
                                        })
                        st.plotly_chart(fig2, use_container_width=True)
                
                # Display topic-specific response times
                if hasattr(df, 'additional_insights') and 'topic_response_times' in df.additional_insights:
                    st.write("### Response Time by Topic")
                    
                    topic_times = df.additional_insights['topic_response_times']
                    
                    # Sort by average response time
                    topic_times = topic_times.sort_values('avg_response_time', ascending=False)
                    
                    # Create bar chart for topic response times
                    fig = px.bar(topic_times, x='topic', y='avg_response_time',
                                title="Average Response Time by Topic",
                                labels={
                                    'topic': 'Topic',
                                    'avg_response_time': 'Average Response Time (seconds)'
                                })
                    fig.update_layout(xaxis={'categoryorder': 'total descending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create scatter plot for response time vs count
                    fig2 = px.scatter(topic_times, x='avg_response_time', y='count', size='count',
                                     hover_data=['topic', 'median_response_time'],
                                     title="Response Time vs. Frequency by Topic",
                                     labels={
                                         'avg_response_time': 'Average Response Time (seconds)',
                                         'count': 'Number of Interactions',
                                         'topic': 'Topic'
                                     })
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Response time data not available.")
        
        with tab5:
            st.subheader("Client Insights")
            
            # Client distribution
            if not df.empty and 'client_name' in df.columns and df['client_name'].notna().sum() > 0:
                try:
                    client_counts = df['client_name'].value_counts().head(10).reset_index()
                    client_counts.columns = ['Client', 'Count']
                    
                    fig = px.bar(client_counts, x='Client', y='Count',
                                title="Top 10 Clients by Interaction Volume")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating client distribution chart: {e}")
                
                # Client type analysis
                if 'client_type' in df.columns and df['client_type'].notna().sum() > 0:
                    try:
                        type_counts = df['client_type'].value_counts().reset_index()
                        type_counts.columns = ['Client Type', 'Count']
                        
                        fig = px.pie(type_counts, values='Count', names='Client Type',
                                    title="Distribution by Client Type")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating client type chart: {e}")
                
                # Client sector analysis
                if 'client_sector' in df.columns and 'response_sentiment' in df.columns:
                    try:
                        sector_df = df.groupby('client_sector').agg({
                            'response_sentiment': 'mean',
                            'thread_id': 'count'
                        }).reset_index()
                        sector_df.columns = ['Sector', 'Avg Sentiment', 'Interaction Count']
                        
                        fig = px.scatter(sector_df, x='Avg Sentiment', y='Interaction Count',
                                        size='Interaction Count', color='Sector', hover_name='Sector',
                                        title="Client Sector Performance")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating sector analysis chart: {e}")
                
                # Client satisfaction analysis if available
                if hasattr(df, 'additional_insights') and 'client_satisfaction' in df.additional_insights:
                    st.write("### Client Satisfaction Analysis")
                    
                    client_satisfaction = df.additional_insights['client_satisfaction']
                    
                    # Sort by average satisfaction
                    client_satisfaction = client_satisfaction.sort_values('avg_satisfaction', ascending=False)
                    
                    # Create bar chart for client satisfaction
                    fig = px.bar(client_satisfaction.head(10), x='client_name', y='avg_satisfaction',
                                title="Top 10 Clients by Satisfaction Score",
                                labels={
                                    'client_name': 'Client',
                                    'avg_satisfaction': 'Average Satisfaction Score'
                                })
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create scatter plot for satisfaction vs response time
                    fig2 = px.scatter(client_satisfaction, x='avg_response_time', y='avg_satisfaction', 
                                     size='count', color='positive_rate',
                                     hover_data=['client_name', 'negative_rate'],
                                     title="Client Satisfaction vs. Response Time",
                                     labels={
                                         'avg_response_time': 'Average Response Time (seconds)',
                                         'avg_satisfaction': 'Average Satisfaction Score',
                                         'positive_rate': 'Positive Reaction Rate',
                                         'count': 'Number of Interactions'
                                     })
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Client data not available.")
                
                # Country analysis
                if 'client_country' in df.columns and df['client_country'].notna().sum() > 0:
                    try:
                        country_counts = df['client_country'].value_counts().reset_index()
                        country_counts.columns = ['Country', 'Count']
                        
                        fig = px.pie(country_counts, values='Count', names='Country',
                                    title="Distribution by Country")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating country chart: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please check your database connection and try again.")

if __name__ == "__main__":
    main()
