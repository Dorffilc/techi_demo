Technician Support System with AI Integration

A Streamlit-based web application designed to assist mechanical and electrical technicians in logging, resolving, and learning from technical issues encountered on site. This project leverages AI for enhanced recommendations, free text processing, and a learning database to provide comprehensive support.

Features
User Authentication: Secure login for technicians to access the system.
Ticket Logging: Technicians can log issues with detailed descriptions, priority levels, and attendance dates.
Ticket Completion: Allows technicians to close tickets with resolution summaries and customer satisfaction ratings.
Dashboard Overview: The Home page provides a summary of key metrics, including the number of tickets attended, average reporting quality, and customer satisfaction scores.
Learning Database: Uses natural language processing (NLP) to analyze past tickets and recommend solutions based on similar historical data.
AI-Powered Recommendations: Integrates OpenAI's ChatGPT to provide intelligent suggestions and assist with troubleshooting.
Cloud-Hosted: Deployed on an AWS Cloud Server for reliable, scalable access.
Technologies Used
Streamlit: Interactive Python library for building web applications.
MongoDB: NoSQL database for storing ticket logs and historical data.
OpenAI's ChatGPT: Natural language processing for enhancing ticket descriptions and providing secondary AI insights.
spaCy: NLP library used for extracting keywords from issue and resolution texts.
Plotly: Used for creating interactive visualizations and diagrams within the application.
AWS Cloud Server: Hosting environment that ensures scalability, security, and high availability.

Installation
Clone the Repository:


git clone https://github.com/your-username/technician-support-system.git
cd technician-support-system

Install Required Packages: Ensure you have Python 3.8+ installed, then install the necessary libraries:

pip install -r requirements.txt

Set Up MongoDB:

Install MongoDB and start the MongoDB service.
Create a database named tech_demo and a collection named techi_tickets.
Set Up Environment Variables: Create a .env file in the project root and add your OpenAI API key:

makefile
Copy code
OPENAI_API_KEY=your_openai_api_key
Run the Application: Start the Streamlit app using the following command:

bash
Copy code
streamlit run app.py
Access the Application: Open your web browser and go to http://localhost:8501 to access the Technician Support System.

Usage
Logging in: Enter your technician credentials to access the system.
Log a Ticket: Go to the "Ticket Logging Page" to describe an issue, select its priority, and submit it for processing.
Complete a Ticket: Navigate to the "Ticket Completion Page" to provide a resolution summary for open tickets.
View Completed Jobs: Use the "Completed Jobs Page" to filter and view resolved tickets by date range.
Get Recommendations: Use the "Recommendation Page" to enter an issue description and receive suggestions from the learning database and ChatGPT.
Contributing
Contributions are welcome! To contribute:

Fork this repository.
Create a new branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add some feature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for more information.

Acknowledgments
Streamlit for providing the platform to build this app.
OpenAI for their powerful language model, ChatGPT.
spaCy for NLP capabilities.
MongoDB for data storage and retrieval.
Plotly for interactive visualizations.
This README should give potential users and contributors a clear understanding of the project, its purpose, and how to get started. Adjust the repository link and other specific details as necessary.
