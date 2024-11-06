import openai
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import os
from openai import OpenAI
import time
import spacy
from collections import Counter,defaultdict

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017/")
db = client["tech_demo"]
collection = db["techi_tickets"]

# Define user credentials
USERNAME = "Hoistech"
PASSWORD = "password123"

## Initialize session state for login status
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Apply conditional CSS based on login status and system theme
if not st.session_state['logged_in']:
    # Gradient background for login screen
    st.markdown(
        """
        <style>
        /* Gradient background for login screen */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background: linear-gradient(to bottom right, royalblue, #FFC72C);
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        /* Login page text color in warm orange */
        .css-1lcbmhc, .css-hxt7ib, .css-17lntkn {
            color: #FFA500 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    # Set main app background to off-white and adjust colors based on system theme
    st.markdown(
        """
        <style>
        /* Set background to off-white for the main app */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background-color: #F5F5F5 !important;
        }

        /* Top bar/header background in Hoistech golden yellow with black text */
        header[data-testid="stHeader"] {
            background-color: #FFC72C !important;
            color: black !important; /* Ensures header text is black */
        }
        header[data-testid="stHeader"] * {
            color: black !important; /* Ensures all header text elements are black */
        }
        
        /* Main text in darker, muted orange/yellow */
        .css-1v3fvcr, .css-hxt7ib, .css-17lntkn, h1, h2, h3, h4, h5, h6, p, div, span {{
        color: {MUTED_ORANGE_YELLOW} !important;

        /* System theme detection */
        @media (prefers-color-scheme: dark) {
            /* Dark theme adjustments */
            .css-1v3fvcr, .css-hxt7ib, .css-17lntkn, h1, h2, h3, h4, h5, h6, p, div, span {
                color: #FFC72C !important;  /* Hoistech Yellow text */
            }
            .stButton>button {
                background-color: #003366 !important; /* Dark Blue button */
                color: #FFC72C !important; /* Yellow button text */
                border: none;
            }
            .stSidebar, .css-1v3fvcr, .css-hxt7ib {
                background-color: #003366 !important; /* Sidebar Dark Blue */
                color: #FFC72C !important; /* Yellow sidebar text */
            }
        }

        @media (prefers-color-scheme: light) {
            /* Light theme adjustments */
            .css-1v3fvcr, .css-hxt7ib, .css-17lntkn, h1, h2, h3, h4, h5, h6, p, div, span {
                color: #003366 !important;  /* Dark Blue text */
            }
            .stButton>button {
                background-color: #FFC72C !important; /* Yellow button */
                color: #003366 !important; /* Dark Blue button text */
                border: none;
            }
            .stSidebar, .css-1v3fvcr, .css-hxt7ib {
                background-color: #E6E6E6 !important; /* Sidebar light grey */
                color: #003366 !important; /* Dark Blue sidebar text */
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
# Initialize session state for login status and page
if 'page' not in st.session_state:
    st.session_state['page'] = "Home"


def extract_keywords(text, top_n=5):
    """
    Extracts the top N keywords from a given text using spaCy NLP.
    """
    doc = nlp(text.lower())
    keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB"]]
    return Counter(keywords).most_common(top_n)


def process_ticket_keywords(ticket_id, issue_text, resolution_text):
    """
    Processes keywords for a specific ticket based on its ai_issue and ai_resolution_summary,
    and updates the ticket with extracted keywords.
    """
    # Extract keywords for issue and resolution
    issue_keywords = extract_keywords(issue_text, top_n=5)
    resolution_keywords = extract_keywords(resolution_text, top_n=5)

    # Update the ticket with extracted keywords in MongoDB
    collection.update_one(
        {"_id": ticket_id},
        {"$set": {
            "issue_keywords": issue_keywords,
            "resolution_keywords": resolution_keywords
        }}
    )

# Login function
def login():
    st.title("Technician Login")

    # Input fields for username and password
    username = st.text_input("Username", key="user")
    password = st.text_input("Password", type="password", key="pass")

    # Login button
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.success("Logged in successfully!")
            # Set session state for login status and default page
            st.session_state['logged_in'] = True
            st.session_state['page'] = "Home"
            st.session_state['username'] = username
            st.rerun()
        else:
            st.error("Invalid username or password")


# Main application logic
def main():
    if st.session_state.get('logged_in'):
        # Sidebar navigation
        st.sidebar.title("Navigation")
        st.sidebar.write(f"Welcome, {st.session_state['username']}!")

        # Display navigation options in sidebar
        page = st.sidebar.radio("Go to Page", ["Home","User Instructions", "Ticket Logging Page", "Ticket Completion Page", "Completed Jobs", "Recommendation Page", "Flow Diagram"])

        # Update the selected page in session state
        st.session_state['page'] = page

        # Logout option
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

        # Display the selected page content
        if st.session_state['page'] == "Home":
            show_home()
        elif st.session_state['page'] == "Ticket Logging Page":
            show_ticket_logging_page()
        elif st.session_state['page'] == "Ticket Completion Page":
            show_ticket_completion_page()
        elif st.session_state['page'] == "Completed Jobs":
            show_complete_jobs_page()
        elif st.session_state['page'] == "Recommendation Page":
            show_recommendation_page()
        elif st.session_state['page'] == "Flow Diagram":
            show_flow_diagram_page()
        elif st.session_state['page'] == "User Instructions":
            show_user_instructions()
    else:
        # Show login page if not logged in
        login()


# Home page function with Dial Gauges
def show_home():
    st.title("Technician Dashboard")


    # Filter tickets for the logged-in technician and current month
    current_month = datetime.now().month
    current_year = datetime.now().year
    technician = st.session_state.get("username")

    # Get the start of the current month as epoch timestamp
    current_month_start_datetime = datetime(current_year, current_month, 1)
    current_month_start_epoch = int(time.mktime(current_month_start_datetime.timetuple()))

    tickets = list(collection.find({
        "technician": technician,
        "attendance_date": {"$gte": current_month_start_epoch}
    }))

    # Data calculations for metrics
    total_tickets = len(tickets)

    # Calculate average reporting quality
    total_reporting_quality = 0
    reporting_quality_count = 0

    for ticket in tickets:
        reporting_quality = ticket.get('reporting_quality')
        if reporting_quality is not None:
            total_reporting_quality += reporting_quality
            reporting_quality_count += 1

    avg_reporting_quality = (
        total_reporting_quality / reporting_quality_count
    ) if reporting_quality_count > 0 else 0

    # Placeholder value for Customer Satisfaction (since the user didn't ask to update it)
    avg_customer_satisfaction = 90  # Assuming a default value; adjust as needed

    # Calculate average time taken (in minutes) by the difference between attendance_date and created_date
    total_time_taken = 0
    time_taken_count = 0
    avg_time_taken = 0

    # Creating the 2x2 grid of gauges using make_subplots with domain type
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}],
               [{'type': 'domain'}, {'type': 'domain'}]],
    )

    # Define each gauge as a separate Indicator
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=total_tickets,
        title={'text': "Tickets"},
        gauge={'axis': {'range': [0, 50]},
               'bar': {'color': "darkblue"}}
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=avg_reporting_quality,
        title={'text': "Quality"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkgreen"}}
    ), row=1, col=2)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=avg_customer_satisfaction,
        title={'text': "Satisfaction"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "purple"}}
    ), row=2, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=avg_time_taken,
        title={'text': "Time (min)"},
        gauge={'axis': {'range': [0, 120]},
               'bar': {'color': "orange"}}
    ), row=2, col=2)

    # Set the background colors to off-white
    fig.update_layout(
        plot_bgcolor="#F5F5F5",  # Set plot (data area) background to off-white
        paper_bgcolor="#F5F5F5",  # Set overall chart background to off-white
        margin=dict(t=50, b=50, l=50, r=50)  # Optional: Set margins for better spacing
    )

    # Adjust layout for a compact display
    fig.update_layout(height=500, width=800, margin=dict(t=50, b=0, l=0, r=0))

    # Display the gauges on the Streamlit page
    st.plotly_chart(fig, use_container_width=True)

    # Displaying a table for most common issues and resolutions
    st.subheader("Most Common Issues and Resolutions")
    issues_df = pd.DataFrame(tickets)
    if not issues_df.empty:
        common_issues = issues_df.groupby("issue")["resolution_summary"].agg(
            lambda x: x.mode()[0] if len(x) > 0 else ""
        ).reset_index()
        common_issues.columns = ["Most Common Issues", "Resolution"]
        st.dataframe(common_issues)
    else:
        st.info("No tickets attended this month.")


# Ticket logging page function
def show_ticket_logging_page():
    st.title("Ticket Logging Page")

    # MongoDB connection setup
    client = MongoClient("mongodb://localhost:27017/")
    db = client["tech_demo"]
    collection = db["techi_tickets"]

    # Ticket logging form
    with st.form("ticket_form"):
        issue = st.text_area("Issue Description", height=150)
        system_id = st.text_input("System ID")
        priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        attendance_date = st.date_input("Date of Attendance", value=datetime.today())
        submitted = st.form_submit_button("Submit")

        attendance_datetime = datetime.combine(attendance_date, datetime.min.time())
        attendance_epoch = int(time.mktime(attendance_datetime.timetuple()))

    if submitted:
        quality = int(Chat_GPT_determine_quality(issue).content)
        ai_corrected = process_and_save_text("issue", issue, system_id)

        st.write(f"Report Quality Score: {quality}")
        # Validate required fields
        if not issue or not system_id:
            st.error("Please fill in all the required fields.")
        else:
            # Prepare the ticket data
            ticket_data = {
                "issue": issue,
                "ai_issue": ai_corrected if ai_corrected else "None",  # Save the corrected text or "None
                "system_id": system_id,
                "priority": priority,
                "technician": st.session_state.get("username"),
                "status": "Open",
                "created_date": int(time.time()),  # Current epoch time
                "attendance_date": attendance_epoch,
                "reporting_quality": quality,
                "customer_satisfaction": 0,

                "resolution_summary": '',
                "time_taken": 0
            }

            # Insert ticket data into MongoDB
            try:
                if quality < 20 or not ai_corrected:
                    st.error("The text could not be processed, please try and explain the issue in a different way.")
                else:
                    collection.insert_one(ticket_data)
                    st.success("Ticket submitted successfully!")

                # Display a "Processing Data" message
                with st.spinner("Processing data..."):
                    time.sleep(2)  # Simulate processing time

                    # Step 1: Send ticket details to ChatGPT for probable issue
                    probable_issue = query_chatgpt_for_issue(issue, system_id)

                    # Display the probable issue and similar tickets
                    st.write("### Probable Issue Identified:")
                    st.write(probable_issue.content)

            except Exception as e:
                st.error(f"An error occurred while submitting the ticket: {e}")


def query_chatgpt_for_issue(issue, system_id):
    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are a troubleshooting assistant for machines, specifically lifting equipment."
                        " Pay attention to the details and provide accurate responses. "
                        " Furthermore, you are also a safety expert for lifting equipment in Ireland,"
                        " when asked a question, also consider the specific safety aspects and regulatory"
                        " requirements in Ireland for the specific job."
                        " Always phrase your answers as suggestions and not as definitive statements."
                        " The person you are communicating with is a technician logging a ticket for a machine issue."},
            {
                "role": "user",
                "content": f"{issue} for system ID {system_id}"
            }
        ]
    )
    probable_issue = completion.choices[0].message
    return probable_issue


def Chat_GPT_determine_quality(description):
    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Your role is to determine the quality of a report based on the given"
                        " description and assign a score out of 100 based on the quality of the report."
                        " Only return a single number as the score out of 100."},
            {
                "role": "user",
                "content": f"{description}"
            }
        ]
    )
    probable_issue = completion.choices[0].message
    return probable_issue


def process_and_save_text(variable_name: str, text: str, ticket_id: str):
    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Your role is to correct the spelling and improve the context of the given text."
                        " Keep in mind this data processed will be subsequently used in a learning database, if "
                        "you cannot make sense of the text return a only the word None"},
            {
                "role": "user",
                "content": f"{text}"
            }
        ]
    )
    corrected_text = completion.choices[0].message
    # Create a dynamic field name for MongoDB
    ai_field_name = f"ai_{variable_name}"

    # Pass text to ChatGPT for contextual correction

    # Save the corrected text in MongoDB under the dynamic field name
    try:
        if corrected_text == "None":
            return None
        else:
            return corrected_text.content

    except Exception as e:
        st.warning(f"Error processing text with ChatGPT: {e}")
        return None


# New function for the Ticket Completion Page
def show_ticket_completion_page():
    st.title("Ticket Completion Page")

    technician = st.session_state.get("username")

    # Fetch open tickets for the technician
    open_tickets = list(collection.find({
        "technician": technician,
        "status": "Open"
    }))

    if not open_tickets:
        st.info("No open tickets to complete.")
        return

    # Display a selectbox to choose a ticket to complete
    ticket_options = [f"Ticket ID: {ticket['_id']} - Issue: {ticket['issue']}" for ticket in open_tickets]
    selected_ticket = st.selectbox("Select a ticket to complete", ticket_options)

    # Find the selected ticket
    ticket_index = ticket_options.index(selected_ticket)
    ticket_to_complete = open_tickets[ticket_index]

    # Ticket completion form
    with st.form("completion_form"):
        resolution_summary = st.text_area("Resolution Summary", height=150)
        customer_satisfaction = st.slider("Customer Satisfaction Rating (0-100)", min_value=0, max_value=100, value=80)
        submitted = st.form_submit_button("Complete Ticket")

    if submitted:
        # Validate required fields
        summary_quality = int(Chat_GPT_determine_quality(resolution_summary).content)
        if summary_quality < 20:
            st.error("The resolution summary is not of sufficient quality, please try again and explain the "
                     "resolution more broadly.")
        else:
            ai_completed_text = process_and_save_text("resolution_summary", resolution_summary, ticket_to_complete["_id"])
            if not resolution_summary:
                st.error("Please provide a resolution summary.")
            else:
                # Ensure 'created_date' is in epoch time (int)
                created_date = ticket_to_complete["created_date"]
                if isinstance(created_date, datetime):
                    created_date_epoch = int(created_date.timestamp())
                elif isinstance(created_date, int):
                    created_date_epoch = created_date
                else:
                    st.error("Unexpected data type for 'created_date'")
                    return

                # Calculate 'time_taken'
                time_taken = int(time.time()) - created_date_epoch

                # Update the ticket in the database
                try:
                    collection.update_one(
                        {"_id": ticket_to_complete["_id"]},
                        {"$set": {
                            "resolution_summary": resolution_summary,
                            "customer_satisfaction": customer_satisfaction,
                            "ai_resolution_summary": ai_completed_text,
                            "status": "Closed",
                            "time_taken": time_taken
                        }}
                    )
                    st.success("Ticket completed successfully!")

                    process_ticket_keywords(ticket_to_complete["_id"], ticket_to_complete["ai_issue"], ai_completed_text)

                except Exception as e:
                    st.error(f"An error occurred while updating the ticket: {e}")


# Function to display the "Complete Jobs" page
# Function to show completed jobs with text-wrapping for long descriptions
def show_complete_jobs_page():
    st.title("Complete Jobs")

    # Date range input for filtering completed jobs
    st.subheader("Select Date Range")
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    end_date = st.date_input("End Date", value=datetime.now())

    # Ensure the end date is after the start date
    if start_date > end_date:
        st.error("End date must be after the start date.")
        return

    # Convert dates to datetime objects for querying
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())

    # Fetch completed jobs within the selected date range
    completed_jobs = list(collection.find({
        "status": "Closed",
        "attendance_date": {"$gte": int(start_datetime.timestamp()), "$lte": int(end_datetime.timestamp())}
    }))

    # Check if there are completed jobs in the date range
    if completed_jobs:
        st.success(f"Found {len(completed_jobs)} completed job(s) in the selected date range.")

        # Convert data to DataFrame for display
        jobs_df = pd.DataFrame(completed_jobs)
        # Filter relevant columns if needed
        jobs_df = jobs_df[["system_id", "issue", "ai_issue", "issue_keywords", "resolution_summary",
                           "ai_resolution_summary", "resolution_keywords", "created_date", "attendance_date", "time_taken"]]

        # Convert timestamp fields to readable dates
        jobs_df["created_date"] = pd.to_datetime(jobs_df["created_date"], unit='s')
        jobs_df["attendance_date"] = pd.to_datetime(jobs_df["attendance_date"], unit='s')

        # Convert lists in 'issue_keywords' and 'resolution_keywords' columns to strings
        jobs_df["issue_keywords"] = jobs_df["issue_keywords"].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x))
        jobs_df["resolution_keywords"] = jobs_df["resolution_keywords"].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x))

        # Inject CSS for text wrapping
        st.markdown(
            """
            <style>
            .dataframe td {
                white-space: normal !important;
                word-wrap: break-word !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Display the completed jobs as a table with wrapped text
        st.dataframe(jobs_df)
    else:
        st.info("No completed jobs found in the selected date range.")


def find_similar_tickets(keyword_list, field="issue_keywords"):
    """
    Finds tickets in the database with similar keywords.

    Parameters:
    - keyword_list (list): The list of keywords to search for.
    - field (str): The field in MongoDB to match keywords against (default is "issue_keywords").

    Returns:
    - list: Tickets that have similar keywords.
    """
    # Construct a MongoDB query to find similar keywords in the specified field
    query = {field: {"$in": keyword_list}}
    similar_tickets = list(collection.find(query))
    return similar_tickets


def cluster_issues_and_resolutions():
    """
    Clusters tickets based on similarity in issue_keywords and resolution_keywords.
    Saves the clusters to the database.
    """
    clusters = defaultdict(list)  # To store clusters with similar tickets

    # Retrieve all tickets with issue and resolution keywords
    tickets = list(collection.find({"issue_keywords": {"$exists": True}, "resolution_keywords": {"$exists": True}}))

    for ticket in tickets:
        issue_keywords = ticket["issue_keywords"]
        resolution_keywords = ticket["resolution_keywords"]

        # Find other tickets with similar issue keywords
        similar_issue_tickets = find_similar_tickets(issue_keywords, field="issue_keywords")
        similar_resolution_tickets = find_similar_tickets(resolution_keywords, field="resolution_keywords")

        # Create a unique cluster ID based on ticket ID or other identifier
        cluster_id = str(ticket["_id"])

        # Store similar tickets in clusters
        clusters[cluster_id] = {
            "ticket_id": ticket["_id"],
            "issue_keywords": issue_keywords,
            "resolution_keywords": resolution_keywords,
            "similar_issues": [t["_id"] for t in similar_issue_tickets],
            "similar_resolutions": [t["_id"] for t in similar_resolution_tickets]
        }

        # Update the ticket with cluster information in MongoDB
        collection.update_one(
            {"_id": ticket["_id"]},
            {"$set": {"issue_cluster": clusters[cluster_id]["similar_issues"],
                      "resolution_cluster": clusters[cluster_id]["similar_resolutions"]}}
        )

    print("Clustering complete. Each ticket now has similar issues and resolutions linked.")
    return clusters


def recommend_resolution(new_issue_text):
    """
    Recommends a resolution for a new ticket based on the clustered historical data.

    Parameters:
    - new_issue_text (str): Description of the new issue.

    Returns:
    - list: Recommended resolutions from similar past issues.
    """
    # Step 1: Extract keywords from the new issue text
    new_issue_keywords = extract_keywords(new_issue_text, top_n=5)

    # Step 2: Find clusters that contain similar keywords
    similar_tickets = find_similar_tickets(new_issue_keywords, field="issue_keywords")

    # Step 3: Collect resolutions from these clusters
    recommended_resolutions = []
    for ticket in similar_tickets:
        if "ai_resolution_summary" in ticket:
            recommended_resolutions.append(ticket["ai_resolution_summary"])

    # Remove duplicate recommendations
    return list(set(recommended_resolutions))


def show_recommendation_page():
    st.title("Ticket Resolution Recommendations")

    # Input for new issue description
    new_issue_text = st.text_area("Describe the issue you're facing", height=150)

    if st.button("Get Recommended Resolutions"):
        if new_issue_text:
            recommendations = recommend_resolution(new_issue_text)

            # Display recommendations
            st.subheader("Recommended Resolutions from learning database")
            if recommendations:
                for res in recommendations:
                    st.write("- ", res)
            else:
                st.write("No similar issues found. Please consult the engineering team.")
            st.subheader("ChatGPT Interaction")
            gpt = query_chatgpt_for_issue(recommendations, "1234")
            st.write(gpt.content)
        else:
            st.warning("Please enter a description of the issue to receive recommendations.")

def show_flow_diagram_page():
    st.title("Application Code Flow Diagram")

    # Define block positions and names for each component with added spacing
    blocks = {
        "Initialize App": (0.1, 0.9),
        "Session State Check": (0.1, 0.75),
        "Apply Theme": (0.1, 0.6),
        "Login Function": (0.4, 0.9),
        "Validate Credentials": (0.4, 0.75),
        "Set Session State": (0.4, 0.6),
        "Main App Logic": (0.4, 0.45),
        "Sidebar Navigation": (0.4, 0.3),
        "Home Page": (0.7, 0.9),
        "Ticket Logging Page": (0.7, 0.75),
        "Ticket Completion Page": (0.7, 0.6),
        "Completed Jobs Page": (0.7, 0.45),
        "Recommendation Page": (0.7, 0.3),
        "ChatGPT Interaction": (1.0, 0.75),
        "Database Interaction": (1.0, 0.6),
        "Logout and Reset": (0.1, 0.45),
    }

    # Define connections (arrows) between blocks
    arrows = [
        ("Initialize App", "Session State Check"),
        ("Session State Check", "Apply Theme"),
        ("Apply Theme", "Login Function"),
        ("Login Function", "Validate Credentials"),
        ("Validate Credentials", "Set Session State"),
        ("Set Session State", "Main App Logic"),
        ("Main App Logic", "Sidebar Navigation"),
        ("Sidebar Navigation", "Home Page"),
        ("Sidebar Navigation", "Ticket Logging Page"),
        ("Sidebar Navigation", "Ticket Completion Page"),
        ("Sidebar Navigation", "Completed Jobs Page"),
        ("Sidebar Navigation", "Recommendation Page"),
        ("Ticket Logging Page", "ChatGPT Interaction"),
        ("Ticket Logging Page", "Database Interaction"),
        ("Ticket Completion Page", "ChatGPT Interaction"),
        ("Ticket Completion Page", "Database Interaction"),
        ("Completed Jobs Page", "Database Interaction"),
        ("Recommendation Page", "ChatGPT Interaction"),
        ("Recommendation Page", "Database Interaction"),
        ("Main App Logic", "Logout and Reset"),
    ]

    # Create Plotly figure with larger canvas
    fig = go.Figure()

    # Add blocks as shapes with annotations for centered text
    for block, (x, y) in blocks.items():
        # Add rectangle shape for each block
        fig.add_shape(
            type="rect",
            x0=x-0.1, y0=y-0.05, x1=x+0.1, y1=y+0.05,
            line=dict(color="RoyalBlue"),
            fillcolor="LightSkyBlue",
        )

        # Add annotation to center text within each block
        fig.add_annotation(
            x=x, y=y, text=block,
            showarrow=False,
            font=dict(size=14, color="black"),  # Increased font size for readability
            xanchor="center", yanchor="middle"
        )

    # Add arrows (connections) between blocks
    for source, target in arrows:
        x0, y0 = blocks[source]
        x1, y1 = blocks[target]
        fig.add_annotation(
            ax=x0, ay=y0, x=x1, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.5, arrowcolor="gray"
        )

    # Update layout for larger display
    fig.update_layout(
        title="Application Code Flow Diagram",
        showlegend=False,
        xaxis=dict(showticklabels=False, zeroline=False, range=[0, 1.2]),
        yaxis=dict(showticklabels=False, zeroline=False, range=[0, 1.0]),
        height=1000,  # Increased height for better readability
        width=1200,   # Increased width for better readability
    )

    # Display the flow diagram on the page
    st.plotly_chart(fig, use_container_width=True)


# Function to show the step-by-step user instruction page
def show_user_instructions():
    st.title("User Instructions")

    st.header("1. Home Page")
    st.markdown("""
        The **Home Page** provides a dashboard overview of your activity...
    """)  # Additional page-specific instructions here...

    st.header("Technologies and AI Integration")
    st.markdown("""
        This application leverages various technologies to deliver a powerful, user-friendly experience. Here’s an overview of the advanced features and technologies used:

        ### 1. AI-Powered Recommendations (ChatGPT)
        The app uses **OpenAI's ChatGPT** model to provide intelligent recommendations and insights. AI powers several parts of the app:
        - **Ticket Logging Assistance**: When logging a new ticket, ChatGPT processes the description text to ensure clarity and provide insights based on the issue.
        - **Recommendation Page**: ChatGPT suggests solutions based on similar past issues, helping technicians solve problems faster and learn from previous cases.
        - **Resolution Quality Scoring**: AI evaluates the quality of each ticket’s resolution summary, ensuring reporting standards are met.

        ### 2. Smart Free Text Filtering
        - **Keyword Extraction**: The application processes ticket descriptions and resolutions to extract key terms using NLP (Natural Language Processing) models.
        - **Similarity Search**: When users enter an issue on the Recommendation Page, AI algorithms analyze keywords and compare them with historical tickets to suggest relevant solutions.

        ### 3. Learning Database Implementation
        The app is designed as a **learning database** that continuously improves based on user inputs:
        - **Keyword Analysis and Clustering**: Each ticket entry is processed for keywords and related tickets are clustered based on similar issues or resolutions.
        - **Continuous Data Enrichment**: Every logged or completed ticket enriches the recommendation engine’s data.
        - **Feedback Integration**: User ratings help gauge which solutions work best, providing valuable feedback for future improvements.

        ### 4. Cloud Hosting on AWS
        The application is hosted on an **AWS Cloud Server**, ensuring high availability, scalability, and secure access. AWS provides the cloud infrastructure that allows the application to handle multiple users efficiently, with low latency and reliable performance.

        ### Technologies Used
        - **Streamlit**: For creating this interactive web application.
        - **MongoDB**: Manages ticket data, including open, completed, and historical tickets.
        - **OpenAI’s ChatGPT**: For natural language understanding, enhancing ticket descriptions, and suggesting solutions.
        - **spaCy**: For NLP processing to extract keywords and group similar issues.
        - **Plotly**: For data visualizations like interactive charts, gauges, and flow diagrams that provide performance insights.
        - **AWS Cloud Server**: Hosts the application, offering secure and scalable cloud infrastructure.
    """)

# Run the main function
if __name__ == "__main__":
    main()
