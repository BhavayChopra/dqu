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

@st.cache_data(ttl=600)  # Cache for 10 minutes
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
            
            # Limit results before grouping to reduce memory usage
            query += " ORDER BY thread_created_on DESC LIMIT 2000"
            
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
                
                # Categorize prompts by topic
                if 'user_prompt' in df.columns:
                    df['topic'] = df['user_prompt'].apply(categorize_prompt_topic)
                
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

# Function to categorize prompts by topic
def categorize_prompt_topic(prompt):
    try:
        prompt = str(prompt).lower() if not pd.isna(prompt) else ""
        
        # Define topic keywords
        topics = {
            'forms': ['form', 'input', 'field', 'submit', 'checkbox', 'radio', 'select'],
            'tables': ['table', 'grid', 'column', 'row', 'cell', 'header'],
            'images': ['image', 'picture', 'photo', 'alt', 'figure'],
            'navigation': ['menu', 'nav', 'navigation', 'link', 'button'],
            'color': ['color', 'contrast', 'background', 'foreground'],
            'keyboard': ['keyboard', 'key', 'shortcut', 'tab', 'focus'],
            'screen reader': ['screen reader', 'jaws', 'nvda', 'voiceover', 'aria'],
            'mobile': ['mobile', 'responsive', 'touch', 'swipe', 'gesture'],
            'pdf': ['pdf', 'document', 'acrobat'],
            'wcag': ['wcag', 'guideline', 'success criterion', 'conformance']
        }
        
        # Check which topics are present in the prompt
        for topic, keywords in topics.items():
            if any(keyword in prompt for keyword in keywords):
                return topic
        
        return 'other'
    except Exception as e:
        # Return a default value if any error occurs
        return 'other'

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
            if 'response_sentiment' in df.columns:
                avg_response_sentiment = df['response_sentiment'].mean()
                st.metric("Avg Response Sentiment", f"{avg_response_sentiment:.2f}")
            else:
                st.metric("Avg Response Sentiment", "N/A")
        with col4:
            if 'feedback' in df.columns:
                feedback_count = df['feedback'].notna().sum()
                st.metric("Feedback Responses", feedback_count)
            else:
                st.metric("Feedback Responses", "N/A")
        
        # Main Visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Trend Analysis", "Content Analysis", "Topic Analysis", "Client Insights"])
        
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
                except Exception as e:
                    st.error(f"Error creating time series plot: {e}")
                
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
            else:
                st.info("Topic categorization not available or insufficient data.")
                
        with tab4:
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
                
                # Country analysis
                if 'client_country' in df.columns and df['client_country'].notna().sum() > 0:
                    try:
                        country_counts = df['client_country'].value_counts().reset_index()
                        country_counts.columns = ['Country', 'Count']
                        
                        fig = px.choropleth(country_counts, locations='Country', 
                                            locationmode='country names',
                                            color='Count', title="Global Distribution",
                                            color_continuous_scale=px.colors.sequential.Plasma)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating country distribution map: {e}")
            else:
                st.info("Client data not available for analysis.")
        
        # Raw Data View
        with st.expander("View Raw Data"):
            try:
                st.dataframe(df[['thread_id', 'user_id', 'client_name', 'thread_created_on', 
                                'user_prompt', 'response', 'reaction', 'topic']].head(100))
            except Exception as e:
                st.error(f"Error displaying raw data: {e}")
                st.dataframe(df.head(100))
    
    except Exception as e:
        st.error(f"An error occurred while loading the dashboard: {e}")
        st.info("Please check your database connection and try again.")

if __name__ == "__main__":
    main()