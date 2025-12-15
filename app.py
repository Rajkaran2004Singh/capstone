import streamlit as st
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import json
import re
import pandas as pd
import zipfile
import os
import shutil
import tempfile
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration and Constants ---

# Use st.secrets for secure API key storage in Streamlit Cloud
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, AttributeError):
    # Fallback for local testing if not using st.secrets
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        # In a real deployed app, this stops execution if the secret isn't set.
        st.error("Gemini API Key not found.")
        st.caption("Please ensure your `GOOGLE_API_KEY` is set in Streamlit Cloud secrets or as an environment variable.")
        st.stop()

# Initialize Gemini Client
try:
    genai.configure(api_key=api_key)
    MODEL_NAME = 'gemini-2.5-flash'
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()


# The detailed prompt for the Gemini model (remains unchanged)
prompt_template = """
You are an expert at reading answer sheet cover pages and marking tables.
Your task is to analyze the provided image and extract specific details.

1.  **Roll Number:** Extract the student's roll number.
2.  **Subject Code:** Extract the subject code of the course.
3.  **Question-wise Marks:** Extract the marks for each question from the evaluation table.
    - If a question's marks are not specified, assign a value of 0.
    - If "NA" or any non-numeric value is written, assign a value of 0.
    - The question number should be the key, and the marks should be the value.
4.  **Calculated Total Marks:** Sum up the marks from all the questions extracted in step 3.

ROI from top left corner of roll number : pt1 = (430,200),pt2 = (580,240)
subject code : pt1 = (157,240),pt2 = (274,270)
Q1 : pt1 = (154,525),pt2 = (212,570)
Q2 : pt1 = (212,525),pt2 = (270,570)
Q3 : pt1 = (270,525),pt2 = (327,570)
Q4 : pt1 = (327,525),pt2 = (384,570)
Q5 : pt1 = (384,525),pt2 = (440,570)
Q6 : pt1 = (440,525),pt2 = (496,570)
Q7 : pt1 = (496,524),pt2 = (551,567)
Q8 : pt1 = (551,520),pt2 = (606,563)
Q9 : pt1 = (606,520),pt2 = (661,563)
Q10 : pt1 = (661,520),pt2 = (716,563)
total marks : pt1 = (716,516),pt2 = (775,560)

Present all the extracted information in a clean, easy-to-parse JSON format.

Example of desired JSON output:
```json
{
  "roll_number": "101903330",
  "subject_code": "UCS 698",
  "question_marks": {
    "Q1": 5,
    "Q2": 8,
    "Q3": 10
  },
  "calculated_total_marks": 23
}
"""


# --- Core Logic Functions ---

@st.cache_data
def convert_df_to_excel(df):
    """Converts a pandas DataFrame to an Excel file in memory."""
    output = BytesIO()
    # Ensure pandas uses the openpyxl engine
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Answer_Sheet_Summary')
    processed_data = output.getvalue()
    return processed_data

def process_single_image(file_name, file_content):
    """
    Calls the Gemini API to extract data from a single image and flattens the result.
    (Function body remains the same as previous)
    """
    flat_data = {'file_name': file_name}
    st.write(f"🔍 Processing **{file_name}**...")

    try:
        img = Image.open(BytesIO(file_content))
        
        with st.spinner(f'Extracting data for {file_name}...'):
            response = model.generate_content([prompt_template, img], stream=False)
            response.resolve()
            response_text = response.text.strip()

        json_match = re.search(r'```json\n(.*?)```', response_text, re.DOTALL)

        if json_match:
            json_string = json_match.group(1)
            try:
                extracted_data = json.loads(json_string)
                flat_data['roll_number'] = extracted_data.get('roll_number', 'N/A')
                flat_data['subject_code'] = extracted_data.get('subject_code', 'N/A')
                flat_data['calculated_total_marks'] = extracted_data.get('calculated_total_marks')

                for q, marks in extracted_data.get('question_marks', {}).items():
                    flat_data[q] = marks
                
                st.success(f"✅ Extracted data for **{flat_data['roll_number']}**.")
            except json.JSONDecodeError:
                flat_data['error'] = 'Invalid JSON output from model.'
                st.error(f"❌ JSON Decode Error for {file_name}.")
        else:
            flat_data['error'] = 'No JSON block found in model response.'
            st.warning(f"⚠️ Could not find a JSON block in the response for {file_name}.")

    except Exception as e:
        st.error(f"An error occurred while processing {file_name}: {e}")
        flat_data['error'] = str(e)

    return flat_data

def process_uploaded_files(uploaded_files):
    """Handles both individual images and a single ZIP file. (Function body remains the same)"""
    all_results = []
    
    if len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith('.zip'):
        zip_file = uploaded_files[0]
        st.info(f"Detected ZIP file: **{zip_file.name}**. Extracting contents...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                zip_path = os.path.join(temp_dir, zip_file.name)
                with open(zip_path, 'wb') as f:
                    f.write(zip_file.read())
                
                extract_dir = os.path.join(temp_dir, "extracted")
                os.makedirs(extract_dir)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    image_files_in_zip = [
                        name for name in zip_ref.namelist() 
                        if not name.startswith('__MACOSX') and name.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ]
                    
                    if not image_files_in_zip:
                        st.warning("No supported image files (.jpg, .jpeg, .png) found inside the ZIP file.")
                        return []
                    
                    for member in image_files_in_zip:
                        zip_ref.extract(member, extract_dir)

                    st.success(f"Extracted {len(image_files_in_zip)} images.")
                    
                    for file_name in sorted(image_files_in_zip):
                        full_path = os.path.join(extract_dir, file_name)
                        display_name = os.path.basename(file_name) 
                        
                        if os.path.exists(full_path) and os.path.isfile(full_path):
                            with open(full_path, 'rb') as f:
                                file_content = f.read()
                            
                            result = process_single_image(display_name, file_content)
                            all_results.append(result)

            except zipfile.BadZipFile:
                st.error(f"Error: **{zip_file.name}** is not a valid zip file.")
            except Exception as e:
                st.error(f"A critical error occurred during zip processing: {e}")
                
    else:
        st.info(f"Processing {len(uploaded_files)} individual file(s)...")
        supported_extensions = ('.jpg', '.jpeg', '.png')
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            if file_name.lower().endswith(supported_extensions):
                file_content = uploaded_file.read()
                result = process_single_image(file_name, file_content)
                all_results.append(result)
            else:
                st.warning(f"Skipping **{file_name}**: Not a supported image type.")

    return all_results

# --- NEW VISUALIZATION FUNCTION ---

def display_visualizations(df, question_cols, total_col):
    """Generates and displays visualizations using Matplotlib/Seaborn."""
    st.header("📈 Class Performance Analysis")
    
    if df.empty:
        st.warning("No data available for visualization.")
        return

    # 1. Average Marks Per Question (Bar Chart)
    try:
        q_avg = df[[q for q in question_cols if q in df.columns]].mean()
        
        if not q_avg.empty:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=q_avg.index, y=q_avg.values, palette="viridis")
            plt.title('Average Marks Secured Per Question', fontsize=16)
            plt.xlabel('Question Number', fontsize=12)
            plt.ylabel('Average Mark', fontsize=12)
            plt.xticks(rotation=0)
            st.pyplot(plt)
            plt.close()
            st.caption("This chart highlights the questions where the class performed best/worst, indicating question difficulty.")
    except Exception as e:
        st.warning(f"Could not generate Average Marks Per Question chart: {e}")

    # 2. Distribution of Total Marks (Histogram/KDE)
    try:
        if total_col in df.columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(df[total_col], kde=True, bins=10, color='skyblue')
            plt.axvline(df[total_col].mean(), color='red', linestyle='--', label=f'Avg: {df[total_col].mean():.2f}')
            plt.title('Distribution of Total Marks', fontsize=16)
            plt.xlabel('Total Marks', fontsize=12)
            plt.ylabel('Number of Students', fontsize=12)
            plt.legend()
            st.pyplot(plt)
            plt.close()
            st.caption("This distribution shows the overall grade curve. The red dashed line indicates the class average.")
    except Exception as e:
        st.warning(f"Could not generate Distribution of Total Marks chart: {e}")

    # 3. Question Difficulty vs. Highest Marks (Scatter/Combined)
    try:
        q_stats = df[[q for q in question_cols if q in df.columns]].agg(['mean', 'max']).T
        q_stats.columns = ['Average', 'Highest']
        
        if not q_stats.empty:
            plt.figure(figsize=(10, 5))
            sns.lineplot(x=q_stats.index, y=q_stats['Highest'], marker='o', label='Highest Mark Secured', color='green')
            sns.lineplot(x=q_stats.index, y=q_stats['Average'], marker='s', label='Average Mark Secured', color='orange')
            plt.title('Highest vs. Average Marks Per Question', fontsize=16)
            plt.xlabel('Question Number', fontsize=12)
            plt.ylabel('Marks', fontsize=12)
            plt.legend()
            st.pyplot(plt)
            plt.close()
            st.caption("This plot compares the best possible score with the class average, showing the gap in performance.")
    except Exception as e:
        st.warning(f"Could not generate Question Stats chart: {e}")

# --- DISPLAY & DOWNLOAD FUNCTION (Modified to include Vis) ---

def display_summary_and_download(all_results):
    """Aggregates results, calculates averages, displays summary, and provides download link."""
    if not all_results:
        return

    df = pd.DataFrame(all_results)
    question_cols = [f'Q{i}' for i in range(1, 11)]
    total_col = 'calculated_total_marks'

    # Convert mark columns to numeric for calculation
    mark_cols = [col for col in question_cols + [total_col] if col in df.columns]
    if mark_cols:
        for col in mark_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    st.header("📊 Results Summary & Analysis")
    
    # 1. Call Visualizations
    display_visualizations(df, question_cols, total_col)
    
    # 2. Display Metrics and Table
    st.subheader("Key Metrics")
    col1, col2 = st.columns(2)
    
    if total_col in df.columns:
        total_avg = df[total_col].mean()
        col1.metric("Overall Class Average", f"{total_avg:.2f} Marks")
        col2.metric("Highest Total Mark", f"{df[total_col].max():.2f} Marks")

    # --- Final DataFrame Preparation ---
    base_cols = ['roll_number', 'subject_code']
    utility_cols = ['file_name', 'error']
    required_columns = base_cols + question_cols + [total_col] + utility_cols
    final_df = df.reindex(columns=[c for c in required_columns if c in df.columns])

    column_mapping = {
        'calculated_total_marks': 'Total Marks',
        'roll_number': 'Roll Number',
        'subject_code': 'Subject Code',
        'file_name': 'File Name',
        'error': 'Error/Notes'
    }
    for i in range(1, 11):
        column_mapping[f'Q{i}'] = f'Ques {i} Marks'

    final_df = final_df.rename(columns=column_mapping)
    
    st.subheader("Extracted Data Table")
    st.dataframe(final_df, use_container_width=True)

    # --- Download Section ---
    st.subheader("Download Results")
    excel_data = convert_df_to_excel(final_df)
    
    st.download_button(
        label="⬇️ Download Full Results as Excel (.xlsx)",
        data=excel_data,
        file_name='answer_sheet_results_summary.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        help="The downloaded file contains all extracted data and a summary."
    )
    st.success("Your summary is ready for download!")


# --- Streamlit App Layout ---

def main():
    st.set_page_config(
        page_title="Gemini Answer Sheet Processor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 AI-Powered Answer Sheet & Mark Extractor")
    st.markdown("Upload answer sheet cover pages (JPG/PNG) or a single ZIP file. Gemini will extract the marks and generate performance charts.")
    
    # Sidebar remains the same...
    st.sidebar.header("Instructions")
    st.sidebar.markdown(
        """
        1.  **Prepare Files:** Ensure your answer sheet images are clear (JPG/PNG).
        2.  **Upload:** Use the uploader below to select multiple images or a single ZIP file.
        3.  **Process:** Click the 'Start Processing' button.
        4.  **Review:** Examine the performance charts and the extracted data table.
        5.  **Download:** Download the final results as an Excel file.
        """
    )
    st.sidebar.info("Powered by Google's Gemini API for advanced image analysis (OCR & Table Extraction).")

    # --- File Uploader ---
    uploaded_files = st.file_uploader(
        "Upload Answer Sheet Images (JPG/PNG) or a single ZIP file",
        type=['png', 'jpg', 'jpeg', 'zip'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"You have uploaded **{len(uploaded_files)}** file(s).")
        
        if st.button("▶️ Start Processing", type="primary"):
            st.markdown("---")
            with st.container():
                # Process the files
                results = process_uploaded_files(uploaded_files)

                # Display the final summary, visualizations, and download link
                if results:
                    st.markdown("---")
                    display_summary_and_download(results)
                else:
                    st.error("No valid data was extracted. Please check your uploaded files.")

if __name__ == '__main__':
    # Set Matplotlib style for better visuals
    plt.style.use('seaborn-v0_8-whitegrid')
    main()
