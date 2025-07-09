# DE Support App

A simple data engineering support tool that scrapes and analyzes website category structures using Python, Streamlit, Playwright, and OpenAI.

---

## Features

- Crawl and extract category/subcategory links from websites
- Optional AI-based filtering with OpenAI + LangChain
- Streamlit interface for interaction
- Export to CSV
- Compatible with Ubuntu or WSL (Windows Subsystem for Linux)

---

---Step 0: add tree, timelog_analyzer,page_inspector and goal_maker to  new folder (modules)
# Step 1: Navigate to your project folder
cd ~/de_support_app  # Or wherever your app is

# Step 2: Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Install Playwright browsers (headless scraping)
python -m playwright install

# Step 5: Run the app
streamlit run app.py
