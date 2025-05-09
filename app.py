from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
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
exploded_df = pd.read_csv('exploded_df.csv')

# Initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Helper function to create word cloud image
def generate_wordcloud(text):
    wc = WordCloud(width=400, height=400, background_color='white').generate(text)
    img = io.BytesIO()
    wc.to_image().save(img, format='PNG')
    img.seek(0)
    return base64.b64encode(img.read()).decode()

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
        ], width=4),

        #wordcloud
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Word Cloud of Reviews"),
                dbc.CardBody([
                    html.Img(id='wordcloud-img', style={'width': '258px', 'height': '258px'})
                ])
            ])
        ], width=4),

        # topics
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Topics"),
                dbc.CardBody(
                    id='topics-card'
                )
            ])
        ])
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
        color_discrete_map={'Negative': 'red', 'Neutral': 'grey', 'Positive': 'green'}
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

    # Create the donut chart
    fig = go.Figure(go.Pie(
        values=[sentiment_score_percent, 100 - sentiment_score_percent],
        labels=["Sentiment", "Remaining"],
        hole=0.8,
        marker=dict(colors=["#28a745", "#e0e0e0"]),
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

# Update word cloud image
@callback(
    Output('wordcloud-img', 'src'),
    Input('state-dropdown', 'value'),
    Input('city-dropdown', 'value'),
    Input('review-period-dropdown', 'value')
)
def update_wordcloud(state, city, review_period):
    filtered = df[
        (df['State'] == state) & 
        (df['City'] == city) & 
        (df['review_period'] == review_period)
    ]
    if filtered.empty:
        return None
    text = " ".join(filtered['cleaned_review'].dropna().astype(str))
    if not text.strip():
        text = "No reviews available for this selection."
    encoded_image = generate_wordcloud(text)
    return f'data:image/png;base64,{encoded_image}'

@callback(
    Output('topics-card', 'children'),
    Input('state-dropdown', 'value'),
    Input('city-dropdown', 'value'),
    Input('review-period-dropdown', 'value')
)
def update_topics_card(state,city, review_period):
    filtered = df[
    (df['State'] == state) & 
    (df['City'] == city) & 
    (df['review_period'] == review_period)
    ]
    if filtered.empty:
        return "Not available for this selection."
    
    topic_counts = {}

    for topic in exploded_df['manual_topic'].unique():
        topic_reviews = exploded_df[exploded_df['manual_topic'] == topic]
        positive_reviews = topic_reviews[topic_reviews['manual_sentiment'] == 'Positive'].shape[0]
        negative_reviews = topic_reviews[topic_reviews['manual_sentiment'] == 'Negative'].shape[0]
        neutral_reviews = topic_reviews[topic_reviews['manual_sentiment'] == 'Neutral'].shape[0]

        topic_counts[topic] = {'Positive': positive_reviews, 'Negative': negative_reviews, 'Neutral': neutral_reviews}

    topic_html = []
    for topic, counts in topic_counts.items():
        topic_html.append(
            dbc.Card([
                dbc.CardHeader(topic),
                dbc.CardBody([
                    html.Div(f"Positive Reviews: {counts['Positive']}"),
                    html.Div(f"Negative Reviews: {counts['Negative']}"),
                    html.Div(f"Neutral Reviews: {counts['Neutral']}")
                ])
            ])
        )

# Run app
if __name__ == '__main__':
    app.run(debug=True)
