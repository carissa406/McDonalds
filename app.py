from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
import math
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# Load the data
df = pd.read_csv('cleaned_data.csv')

# Initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Helper function to create word cloud image
# def generate_wordcloud(text):
#     wc = WordCloud(width=400, height=400, background_color='white').generate(text)
#     img = io.BytesIO()
#     wc.to_image().save(img, format='PNG')
#     img.seek(0)
#     return base64.b64encode(img.read()).decode()

#calculating sentiment score
sentiment_points = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['sentiment_points'] = df['manual_sentiment'].map(sentiment_points)

# Layout

app.layout = dbc.Container([
    html.H1("McDonald's Review Sentiment Dashboard", className="my-3"),
    html.Hr(),

    # Controls
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Controls"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select a state:"),
                            dcc.Dropdown(df['State'].unique(), value=df['State'].iloc[0], id='state-dropdown')
                        ], width=4),

                        dbc.Col([
                            html.Label("Select a city:"),
                            dcc.Dropdown(id='city-dropdown')
                        ], width=4),

                        dbc.Col([
                            html.Label("Select a period:"),
                            dcc.Dropdown(options=[
                                {'label':p, 'value':p} for p in df['review_period'].unique()], 
                                value=df['review_period'].iloc[0],
                                id='review-period-dropdown')
                        ], width=4),
                    ])
                ])
            ])
        ])
    ]),

     #sentiment score
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sentiment Score"),
                dbc.CardBody([
                    html.H4(className="card-title"),
                    # Add this ID so that the callback can update it dynamically
                    html.Div(id='sentiment-score')
                ])
            ])
        ], width=2),

    # topics
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Topics"),
                dbc.CardBody(
                    id='topics-card')
            ])
        ], width=2),

        #sentiment trend graph
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Sentiment Trend'),
                dbc.CardBody([
                    html.Div(id="sentiment-trend-graph")
                ])
            ])
        ]),
    ]),

    # Three bar charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Rating Distribution"),
                dbc.CardBody([
                    dcc.Graph(id='rating-graph')
                ])
            ])
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Review Length Distribution"),
                dbc.CardBody([
                    dcc.Graph(id='length-graph')
                ])
            ])
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sentiment Distribution"),
                dbc.CardBody([
                    dcc.Graph(id='sentiment-graph')
                ])
            ])
        ], width=4),
    ]),

], fluid=True)

# Callbacks
# Update city dropdown based on selected state
@callback(
    Output('city-dropdown', 'options'),
    Output('city-dropdown', 'value'),
    Input('state-dropdown', 'value'),
)
def update_city_dropdown(state):
    filtered = df[df['State'] == state]
    cities = filtered['City'].unique()
    return [{'label': c, 'value': c} for c in cities], cities[0]

#rating graph
@callback(
    Output('rating-graph', 'figure'),
    Input('state-dropdown', 'value'),
    Input('city-dropdown', 'value'),
    Input('review-period-dropdown', 'value')
)
def update_rating_graph(state, city, review_period):
    filtered = df[
        (df['State'] == state) & 
        (df['City'] == city) & 
        (df['review_period'] == review_period)
    ]
    if filtered.empty:
        return {
        'data': [],
        'layout': {
            'title': 'No data available for this selection.',
            'xaxis': {'title': 'Metric'},
            'yaxis': {'title': 'Count'}
        }
    }
    fig = px.histogram(filtered, x='rating', nbins=20)
    fig.update_layout(xaxis_title='Rating', yaxis_title='Count')
    return fig

#length graph
@callback(
    Output('length-graph', 'figure'),
    Input('state-dropdown', 'value'),
    Input('city-dropdown', 'value'),
    Input('review-period-dropdown', 'value')
)
def update_length_graph(state, city, review_period):
    filtered = df[
        (df['State'] == state) & 
        (df['City'] == city) & 
        (df['review_period'] == review_period)
    ]
    if filtered.empty:
        return {
        'data': [],
        'layout': {
            'title': 'No data available for this selection.',
            'xaxis': {'title': 'Metric'},
            'yaxis': {'title': 'Count'}
        }
    }
    fig = px.histogram(filtered, x='review_length', nbins=20)
    fig.update_layout(xaxis_title='Review Length', yaxis_title='Count')
    return fig

#sentiment graph
@callback(
    Output('sentiment-graph', 'figure'),
    Input('state-dropdown', 'value'),
    Input('city-dropdown', 'value'),
    Input('review-period-dropdown', 'value')
)
def update_sentiment_graph(state, city, review_period):
    filtered = df[
        (df['State'] == state) & 
        (df['City'] == city) & 
        (df['review_period'] == review_period)
    ]
    if filtered.empty:
        return {
        'data': [],
        'layout': {
            'title': 'No data available for this selection.',
            'xaxis': {'title': 'Metric'},
            'yaxis': {'title': 'Count'}
        }
    }
    fig = px.histogram(
        filtered,
        y='manual_sentiment',
        color='manual_sentiment',
        category_orders={'manual_sentiment': ['Negative', 'Neutral', 'Positive']},
        color_discrete_map={'Negative': '#EF5350', 'Neutral': 'grey', 'Positive': '#4CAF50'}
    )
    fig.update_layout(showlegend=False, xaxis_title='Count', yaxis_title='Sentiment')
    return fig

#update sentiment score
@callback(
    Output('sentiment-score', 'children'),  # Output to the HTML element displaying the score
    Input('state-dropdown', 'value'),
    Input('city-dropdown', 'value'),
    Input('review-period-dropdown', 'value')
)
def update_sentiment_score(state, city, review_period):
    # Filter data based on dropdown selections
    filtered = df[
        (df['State'] == state) & 
        (df['City'] == city) & 
        (df['review_period'] == review_period)
    ]
    if filtered.empty:
        return "Not available for this selection."
    
    # Calculate total points and max points
    total_points = filtered['sentiment_points'].sum()
    max_points = len(filtered) * 2  # 2 points max per review (Positive = 2, Neutral = 1, Negative = 0)
    
    # Calculate sentiment score percentage
    sentiment_score_percent = (total_points / max_points) * 100
    
    # Round the percentage to the nearest integer and remove decimals
    sentiment_score_percent = int(round(sentiment_score_percent))

    # Set the color based on sentiment score
    if sentiment_score_percent > 75:
        color = "#28a745"  # Green for high score
    elif sentiment_score_percent > 50:
        color = "#ffc107"  # Yellow for neutral score
    else:
        color = "#dc3545"  # Red for low score

    # Create the donut chart
    fig = go.Figure(go.Pie(
        values=[sentiment_score_percent, 100 - sentiment_score_percent],
        labels=["Sentiment", "Remaining"],
        hole=0.8,
        marker=dict(colors=[color, "#e0e0e0"]),
        textinfo="none",
        hoverinfo="label+percent"
    ))

    # Add the sentiment score as text inside the donut chart
    fig.add_annotation(
        text=f"{sentiment_score_percent}",  # Display sentiment score percentage
        font=dict(size=24, color="black"),  # Font size and color for the score
        showarrow=False,  # No arrow, just the text
        align="center",  # Center the text
        x=0.5,  # Position the text at the center
        y=0.5,  # Position the text at the center
        xanchor="center",
        yanchor="middle"
    )

     # Update layout of the pie chart to make it look like a donut
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        height=250
    )

    return dcc.Graph(figure=fig)

#topics card
@callback(
    Output('topics-card', 'children'),
    Input('state-dropdown', 'value'),
    Input('city-dropdown', 'value'),
    Input('review-period-dropdown', 'value')
)
def update_topics_card(state, city, review_period):
    filtered = df[
        (df['State'] == state) & 
        (df['City'] == city) & 
        (df['review_period'] == review_period)
    ]
    if filtered.empty:
        return "Not available for this selection."
    
    # Initialize dictionary to hold topic sentiment scores and counts
    topic_scores = {}

    # Loop through each unique topic
    for topic in df['manual_topic'].unique():
        # Filter reviews for the current topic
        if topic == "Other":
            continue
        topic_reviews = filtered[filtered['manual_topic'] == topic]
        
        # Calculate sentiment counts for this topic
        positive_reviews = topic_reviews[topic_reviews['manual_sentiment'] == 'Positive'].shape[0]
        negative_reviews = topic_reviews[topic_reviews['manual_sentiment'] == 'Negative'].shape[0]
        neutral_reviews = topic_reviews[topic_reviews['manual_sentiment'] == 'Neutral'].shape[0]

        total_reviews = positive_reviews + negative_reviews + neutral_reviews
        if total_reviews > 0:
            sentiment_score = (positive_reviews * 2 + neutral_reviews) / (total_reviews * 2) * 100
        else:
            sentiment_score = 0  # If no reviews, assign 0 score

        # Add sentiment score to dictionary
        topic_scores[topic] = {
            'sentiment_score': sentiment_score, 
            'positive_reviews': positive_reviews, 
            'negative_reviews': negative_reviews, 
            'neutral_reviews': neutral_reviews
        }

    # Create topic score cards with a circle for each topic
    topic_html = []
    
    for topic, scores in topic_scores.items():
        rounded_score = math.ceil(scores['sentiment_score'])
        
        # Determine color based on sentiment score
        if rounded_score > 75:
            color = "#28a745"  # Green for high score
        elif rounded_score > 50:
            color = "#ffc107"  # Yellow for neutral score
        else:
            color = "#dc3545"  # Red for low score
        
        # Create a circle with the sentiment score inside
        topic_html.append(
                html.Div(
                    style={
                        "display": "flex", 
                        "align-items": "center",  # Center the topic and circle horizontally
                        "margin-bottom": "10px"  # Space between items
                    },
                    children=[
                        html.Div(
                            topic,
                            style={
                                "font-size": "16px", 
                                "font-weight": "bold", 
                                "width": "150px"
                            }
                        ),
                        html.Div(
                            f"{rounded_score}",
                            style={
                                "display": "flex",
                                "justify-content": "center",
                                "align-items": "center",
                                "width": "50px",  # Set size of circle
                                "height": "50px",
                                "border-radius": "50%",  # Make it a circle
                                "background-color": color,  # Set color dynamically
                                "color": "white",  # Text color
                                "font-size": "14px",  # Adjust font size
                                "font-weight": "bold",
                            },
                        ),
                    ]
                )
        )

    # Wrap the topic circles in a dbc.Row to display them in a single row
    return dbc.Row(topic_html)


@callback(
    Output('sentiment-trend-graph', 'children'),
    Input('state-dropdown', 'value'),
    Input('city-dropdown', 'value'),
)
def generate_sentiment_trend_graph(state, city):
    # Define the correct order for review_period
    period_order = [
        "Past 6 Months", "1 Year Ago", "2 Years Ago", "3 Years Ago",
        "4 Years Ago", "5 Years Ago", "6 Years Ago", "7 Years Ago",
        "8 Years Ago", "9 Years Ago", "10 Years Ago"
    ]

    # Filter for selected state and city
    filtered = df[(df['State'] == state) & (df['City'] == city)]
    if filtered.empty:
        return html.Div("No data available for this selection.")

    # Initialize a dictionary for scores
    trend_data = {}

    # Group by review_period and compute sentiment scores
    for period, group in filtered.groupby('review_period'):
        pos = group[group['manual_sentiment'] == 'Positive'].shape[0]
        neu = group[group['manual_sentiment'] == 'Neutral'].shape[0]
        neg = group[group['manual_sentiment'] == 'Negative'].shape[0]
        total = pos + neu + neg

        score = (pos * 2 + neu) / (total * 2) * 100 if total > 0 else 0
        trend_data[period] = round(score, 2)

    # Create DataFrame from full list to ensure correct order and fill missing with NaN
    trend_df = pd.DataFrame({
        "review_period": period_order,
        "sentiment_score": [trend_data.get(p, None) for p in period_order]
    })

    # Create the line chart
    fig = px.line(
        trend_df,
        x='review_period',
        y='sentiment_score',
        markers=True,
        labels={'review_period': 'Review Period', 'sentiment_score': 'Sentiment Score (%)'},
        title=f"Overall Sentiment Score Trend in {city}, {state}",
        category_orders={"review_period": period_order}
    )
    fig.update_layout(
        height=400,
        xaxis_title="Review Period",
        yaxis_title="Sentiment Score (%)",
        yaxis_range=[0, 100]
    )

    return dcc.Graph(figure=fig)

# Update word cloud image
# @callback(
#     Output('wordcloud-img', 'src'),
#     Input('state-dropdown', 'value'),
#     Input('city-dropdown', 'value'),
#     Input('review-period-dropdown', 'value')
# )
# def update_wordcloud(state, city, review_period):
#     filtered = df[
#         (df['State'] == state) & 
#         (df['City'] == city) & 
#         (df['review_period'] == review_period)
#     ]
#     if filtered.empty:
#         return None
#     text = " ".join(filtered['cleaned_review'].dropna().astype(str))
#     if not text.strip():
#         text = "No reviews available for this selection."
#     encoded_image = generate_wordcloud(text)
#     return f'data:image/png;base64,{encoded_image}'


# Run app
if __name__ == '__main__':
    app.run(debug=True)
