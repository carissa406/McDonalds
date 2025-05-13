# McDonald's Review Sentiment Dashboard

![Dashboard](https://github.com/carissa406/McDonalds/blob/main/dashboard.PNG)

## Overview
This dashboard allows users to explore and analyze sentiment trends of customer reviews for McDonald's across different states, cities, and review periods. The dashboard provides key insights into the sentiment distribution, rating distribution, and review length distribution through interactive graphs.

### **Please see my reviews.ipynb file or the PowerPoint Presentation for more in depth analysis.**

The application uses sentiment analysis to categorize customer reviews into positive, negative, and neutral sentiments, and visualizes these insights in a clean, user-friendly layout.

## Key Features
- **Dynamic Filters**: Users can filter reviews based on:
  - State
  - City
  - Review Period (e.g., specific time ranges)

- **Sentiment Analysis**: Display of sentiment scores for different topics within the reviews, with color-coded circles indicating the sentiment levels (e.g., red for negative, yellow for neutral, green for positive).

- **Interactive Visualizations**:
  - **Rating Distribution**: Bar chart showing the distribution of ratings.
  - **Review Length Distribution**: Histogram displaying the distribution of review lengths.
  - **Sentiment Distribution**: Pie chart showing the overall distribution of positive, negative, and neutral sentiments across reviews.

- **Responsive Design**: The dashboard layout adapts to different screen sizes, ensuring a smooth user experience on both desktop and mobile devices.

## Technologies Used
- **Dash**: For creating the interactive dashboard.
- **Plotly**: For rendering interactive graphs and charts.
- **Pandas**: For data manipulation and analysis.
- **Bootstrap (Dash Bootstrap Components)**: For responsive design and UI components.
- **Python**: For backend data processing and sentiment analysis.

## Installation Instructions
To run this project locally, follow these steps:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/carissa406/McDonalds.git
   cd mcdonalds-review-dashboard
2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install the required dependencies
   ```bash
   pip install -r requirements.txt
4. Run the Dash app
   ```bash
   python app.py
5. Open the dashboard in your browser

## Data
This project uses a dataset containing customer reviews for McDonald's. Obtained from Kaggle: https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews
The dataset included fields such as 
- store_address
- review_time
- review
- star rating
The data was then preprocessed. review_time was transformed to review_period. The reviews were cleaned of stop words and most common topics were identified. Sentiment was classified based on star rating.

## Usage
- Select a state, city, and review period from the dropdown menus to filter the reviews as see the sentiment scores of various topics as well as rating distribution, review length distribution, and sentiment distribution.
- The sentiment scores are scored based on the following formula:
  - Negative Sentiment is scored as 0 points
  - Neutral Sentiment is scored as 1 point
  - Positive Sentiment is scored as 2 points
      - Final Score = Total Points Scored / Maximum Possible Points
- Sentiment scores are color coded circles based on score.
