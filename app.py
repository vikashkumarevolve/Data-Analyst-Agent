import json
import tempfile
import csv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from phi.model.openai import OpenAIChat
from phi.agent.duckdb import DuckDbAgent
from phi.tools.pandas import PandasTools
from dotenv import load_dotenv
import os

# Load the API key from the .env file
load_dotenv()

# Streamlit app
st.title("📊 Data Analyst Agent with Visualization")

# Fetch OpenAI API Key from environment variables
openai_key = os.getenv("OPENAI_API_KEY")

if openai_key:
    st.session_state.openai_key = openai_key
    st.success("API key loaded from .env file!")
else:
    st.error("Please provide your OpenAI API key in the .env file to proceed.")

# Ensure chat history is stored in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Keep as is if conversion fails
                    pass
        
        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            # Save the DataFrame to the temporary CSV file with quotes around string fields
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and "openai_key" in st.session_state:
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)
    
    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        st.write("Uploaded Data:")
        st.dataframe(df)  # Use st.dataframe for an interactive table
        
        # Display the columns of the uploaded data
        st.write("Uploaded columns:", columns)
        
        # Configure the semantic model with the temporary file path
        semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                    "path": temp_path,
                }
            ]
        }
        
        # Initialize the DuckDbAgent for SQL query generation
        duckdb_agent = DuckDbAgent(
            model=OpenAIChat(model="gpt-4o-mini", api_key=st.session_state.openai_key),
            semantic_model=json.dumps(semantic_model),
            tools=[PandasTools()],
            markdown=True,
            add_history_to_messages=False,  # Disable chat history for agent itself
            followups=False,  # Disable follow-up queries
            read_tool_call_history=False,  # Disable reading tool call history
            system_prompt="You are an expert data analyst. Generate SQL queries to solve the user's query. Return only the SQL query, enclosed in ```sql ``` and give the final answer.",
        )
        
        # Main query input widget
        user_query = st.text_area("Ask a query about the data:")

        # Add info message about terminal output
        st.info("💡 Check your terminal for a clearer output of the agent's response")

        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner('Processing your query...'):
                        # Get the response from DuckDbAgent
                        response1 = duckdb_agent.run(user_query)

                        # Extract the content from the RunResponse object
                        if hasattr(response1, 'content'):
                            response_content = response1.content
                        else:
                            response_content = str(response1)

                        # Check if response contains any data to visualize
                        if "SELECT" in response_content or "result" in response_content:
                            # Generate a visualization (example: plot the result)
                            st.write("Generated SQL Query: ", response_content)
                            st.write("Visualizing the data...")

                            # Assuming 'response_content' returns a Pandas DataFrame or can be converted to one:
                            try:
                                # Here you might need to actually fetch the data based on the SQL query or mock data
                                # Let's assume the 'df' returned here is the data to visualize
                                st.write(df.head())  # Just displaying a part of the data as a table

                                # Example: Generate a bar plot for two columns 'col1' and 'col2'
                                if 'col1' in df.columns and 'col2' in df.columns:
                                    fig, ax = plt.subplots()
                                    df.plot(kind='bar', x='col1', y='col2', ax=ax)
                                    st.pyplot(fig)

                                # Example of a scatter plot for numerical columns
                                if df.select_dtypes(include=['float64', 'int64']).shape[1] >= 2:
                                    fig, ax = plt.subplots()
                                    df.plot(kind='scatter', x=df.columns[0], y=df.columns[1], ax=ax)
                                    st.pyplot(fig)

                            except Exception as e:
                                st.error(f"Error visualizing data: {e}")
                        else:
                            st.markdown(response_content)
                        
                        # Store the query and response in session history
                        st.session_state.chat_history.append({"query": user_query, "response": response_content})

                except Exception as e:
                    st.error(f"Error generating response from the DuckDbAgent: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")

# **Formatted Chat History View:**
st.write("### Chat History")

# Display the chat history with a conversational, bubble-like format
for i, chat in enumerate(st.session_state.chat_history):
    # Chat bubbles for Queries
    if i % 2 == 0:  # Queries on the left side
        st.markdown(f'<div style="background-color:#f2f2f2;padding:10px;margin:10px 0;border-radius:8px;max-width:60%;float:left;width:auto;">{chat["query"]}</div>', unsafe_allow_html=True)
    else:  # Responses on the right side
        st.markdown(f'<div style="background-color:#d1f7d6;padding:10px;margin:10px 0;border-radius:8px;max-width:60%;float:right;width:auto;">{chat["response"]}</div>', unsafe_allow_html=True)




# import json
# import tempfile
# import csv
# import streamlit as st
# import pandas as pd
# from phi.model.openai import OpenAIChat
# from phi.agent.duckdb import DuckDbAgent
# from phi.tools.pandas import PandasTools
# from dotenv import load_dotenv
# import os

# # Load the API key from the .env file
# load_dotenv()

# # Streamlit app
# st.title("📊 Data Analyst Agent")

# # Fetch OpenAI API Key from environment variables
# openai_key = os.getenv("OPENAI_API_KEY")

# if openai_key:
#     st.session_state.openai_key = openai_key
#     st.success("API key loaded from .env file!")
# else:
#     st.error("Please provide your OpenAI API key in the .env file to proceed.")
        
# # Ensure chat history is stored in session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Function to preprocess and save the uploaded file
# def preprocess_and_save(file):
#     try:
#         # Read the uploaded file into a DataFrame
#         if file.name.endswith('.csv'):
#             df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
#         elif file.name.endswith('.xlsx'):
#             df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
#         else:
#             st.error("Unsupported file format. Please upload a CSV or Excel file.")
#             return None, None, None
        
#         # Ensure string columns are properly quoted
#         for col in df.select_dtypes(include=['object']):
#             df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
#         # Parse dates and numeric columns
#         for col in df.columns:
#             if 'date' in col.lower():
#                 df[col] = pd.to_datetime(df[col], errors='coerce')
#             elif df[col].dtype == 'object':
#                 try:
#                     df[col] = pd.to_numeric(df[col])
#                 except (ValueError, TypeError):
#                     # Keep as is if conversion fails
#                     pass
        
#         # Create a temporary file to save the preprocessed data
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
#             temp_path = temp_file.name
#             # Save the DataFrame to the temporary CSV file with quotes around string fields
#             df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
#         return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
#     except Exception as e:
#         st.error(f"Error processing file: {e}")
#         return None, None, None

# # File upload widget
# uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# if uploaded_file is not None and "openai_key" in st.session_state:
#     # Preprocess and save the uploaded file
#     temp_path, columns, df = preprocess_and_save(uploaded_file)
    
#     if temp_path and columns and df is not None:
#         # Display the uploaded data as a table
#         st.write("Uploaded Data:")
#         st.dataframe(df)  # Use st.dataframe for an interactive table
        
#         # Display the columns of the uploaded data
#         st.write("Uploaded columns:", columns)
        
#         # Configure the semantic model with the temporary file path
#         semantic_model = {
#             "tables": [
#                 {
#                     "name": "uploaded_data",
#                     "description": "Contains the uploaded dataset.",
#                     "path": temp_path,
#                 }
#             ]
#         }
        
#         # Initialize the DuckDbAgent for SQL query generation
#         duckdb_agent = DuckDbAgent(
#             model=OpenAIChat(model="gpt-4o-mini", api_key=st.session_state.openai_key),
#             semantic_model=json.dumps(semantic_model),
#             tools=[PandasTools()],
#             markdown=True,
#             add_history_to_messages=False,  # Disable chat history for agent itself
#             followups=False,  # Disable follow-up queries
#             read_tool_call_history=False,  # Disable reading tool call history
#             system_prompt="You are an expert data analyst. Generate SQL queries to solve the user's query. Return only the SQL query, enclosed in ```sql ``` and give the final answer.",
#         )
        
#         # Main query input widget
#         user_query = st.text_area("Ask a query about the data:")

#         # Add info message about terminal output
#         st.info("💡 Check your terminal for a clearer output of the agent's response")

#         if st.button("Submit Query"):
#             if user_query.strip() == "":
#                 st.warning("Please enter a query.")
#             else:
#                 try:
#                     # Show loading spinner while processing
#                     with st.spinner('Processing your query...'):
#                         # Get the response from DuckDbAgent
#                         response1 = duckdb_agent.run(user_query)

#                         # Extract the content from the RunResponse object
#                         if hasattr(response1, 'content'):
#                             response_content = response1.content
#                         else:
#                             response_content = str(response1)
                        
#                         # Store the query and response in session history
#                         st.session_state.chat_history.append({"query": user_query, "response": response_content})

#                     # Display the response in Streamlit
#                     st.markdown(response_content)
                
#                 except Exception as e:
#                     st.error(f"Error generating response from the DuckDbAgent: {e}")
#                     st.error("Please try rephrasing your query or check if the data format is correct.")

# # **Formatted Chat History View:**
# st.write("### Chat History")

# # Display the chat history with a conversational, bubble-like format
# for i, chat in enumerate(st.session_state.chat_history):
#     # Chat bubbles for Queries
#     if i % 2 == 0:  # Queries on the left side
#         st.markdown(f'<div style="background-color:#f2f2f2;padding:10px;margin:10px 0;border-radius:8px;max-width:60%;float:left;width:auto;">{chat["query"]}</div>', unsafe_allow_html=True)
#     else:  # Responses on the right side
#         st.markdown(f'<div style="background-color:#d1f7d6;padding:10px;margin:10px 0;border-radius:8px;max-width:60%;float:right;width:auto;">{chat["response"]}</div>', unsafe_allow_html=True)



# import json
# import tempfile
# import csv
# import streamlit as st
# import pandas as pd
# from phi.model.openai import OpenAIChat
# from phi.agent.duckdb import DuckDbAgent
# from phi.tools.pandas import PandasTools
# import re

# # Function to preprocess and save the uploaded file
# def preprocess_and_save(file):
#     try:
#         # Read the uploaded file into a DataFrame
#         if file.name.endswith('.csv'):
#             df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
#         elif file.name.endswith('.xlsx'):
#             df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
#         else:
#             st.error("Unsupported file format. Please upload a CSV or Excel file.")
#             return None, None, None
        
#         # Ensure string columns are properly quoted
#         for col in df.select_dtypes(include=['object']):
#             df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
#         # Parse dates and numeric columns
#         for col in df.columns:
#             if 'date' in col.lower():
#                 df[col] = pd.to_datetime(df[col], errors='coerce')
#             elif df[col].dtype == 'object':
#                 try:
#                     df[col] = pd.to_numeric(df[col])
#                 except (ValueError, TypeError):
#                     # Keep as is if conversion fails
#                     pass
        
#         # Create a temporary file to save the preprocessed data
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
#             temp_path = temp_file.name
#             # Save the DataFrame to the temporary CSV file with quotes around string fields
#             df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
#         return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
#     except Exception as e:
#         st.error(f"Error processing file: {e}")
#         return None, None, None

# # Streamlit app
# st.title("📊 Data Analyst Agent")

# # Sidebar for API keys
# with st.sidebar:
#     st.header("API Keys")
#     openai_key = st.text_input("Enter your OpenAI API key:", type="password")
#     if openai_key:
#         st.session_state.openai_key = openai_key
#         st.success("API key saved!")
#     else:
#         st.warning("Please enter your OpenAI API key to proceed.")

# # File upload widget
# uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# if uploaded_file is not None and "openai_key" in st.session_state:
#     # Preprocess and save the uploaded file
#     temp_path, columns, df = preprocess_and_save(uploaded_file)
    
#     if temp_path and columns and df is not None:
#         # Display the uploaded data as a table
#         st.write("Uploaded Data:")
#         st.dataframe(df)  # Use st.dataframe for an interactive table
        
#         # Display the columns of the uploaded data
#         st.write("Uploaded columns:", columns)
        
#         # Configure the semantic model with the temporary file path
#         semantic_model = {
#             "tables": [
#                 {
#                     "name": "uploaded_data",
#                     "description": "Contains the uploaded dataset.",
#                     "path": temp_path,
#                 }
#             ]
#         }
        
#         # Initialize the DuckDbAgent for SQL query generation
#         duckdb_agent = DuckDbAgent(
#             model=OpenAIChat(model="gpt-4", api_key=st.session_state.openai_key),
#             semantic_model=json.dumps(semantic_model),
#             tools=[PandasTools()],
#             markdown=True,
#             add_history_to_messages=False,  # Disable chat history
#             followups=False,  # Disable follow-up queries
#             read_tool_call_history=False,  # Disable reading tool call history
#             system_prompt="You are an expert data analyst. Generate SQL queries to solve the user's query. Return only the SQL query, enclosed in ```sql ``` and give the final answer.",
#         )
        
#         # Initialize code storage in session state
#         if "generated_code" not in st.session_state:
#             st.session_state.generated_code = None
        
#         # Main query input widget
#         user_query = st.text_area("Ask a query about the data:")
        
#         # Add info message about terminal output
#         st.info("💡 Check your terminal for a clearer output of the agent's response")
        
#         if st.button("Submit Query"):
#             if user_query.strip() == "":
#                 st.warning("Please enter a query.")
#             else:
#                 try:
#                     # Show loading spinner while processing
#                     with st.spinner('Processing your query...'):
#                         # Get the response from DuckDbAgent
               
#                         response1 = duckdb_agent.run(user_query)

#                         # Extract the content from the RunResponse object
#                         if hasattr(response1, 'content'):
#                             response_content = response1.content
#                         else:
#                             response_content = str(response1)
#                         response = duckdb_agent.print_response(
#                         user_query,
#                         stream=True,
#                         )

#                     # Display the response in Streamlit
#                     st.markdown(response_content)
                
                    
#                 except Exception as e:
#                     st.error(f"Error generating response from the DuckDbAgent: {e}")
#                     st.error("Please try rephrasing your query or check if the data format is correct.")