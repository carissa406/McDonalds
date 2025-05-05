from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
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
app = Dash()

# Helper function to create word cloud image
def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    img = io.BytesIO()
    wc.to_image().save(img, format='PNG')
    img.seek(0)
    return base64.b64encode(img.read()).decode()

# Layout
app.layout = html.Div([
    html.H1("McDonald's Review Sentiment Dashboard"),
    html.Hr(),

    html.Div([
        html.Label("Select a state:"),
        dcc.Dropdown(df['State'].unique(), value=df['State'].iloc[0], id='state-dropdown'),

        html.Label("Select a city:"),
        dcc.Dropdown(id='city-dropdown'),

        html.Label("Select a metric:"),
        dcc.RadioItems(
            options=[
                {'label': 'Rating', 'value': 'rating'},
                {'label': 'Review Length', 'value': 'review_length'},
                {'label': 'Sentiment', 'value': 'manual_sentiment'}
            ],
            value='rating',
            id='metric-radio'
        )
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        dcc.Graph(id='main-graph')
    ], style={'width': '68%', 'display': 'inline-block', 'paddingLeft': '2%'}),

    html.Div([
        html.H3("Word Cloud of Reviews"),
        html.Img(id='wordcloud-img', style={'width': '100%', 'height': 'auto'})
    ])
])

# Callbacks

# Update city dropdown based on selected state
@callback(
    Output('city-dropdown', 'options'),
    Output('city-dropdown', 'value'),
    Input('state-dropdown', 'value')
)
def update_city_dropdown(state):
    filtered = df[df['State'] == state]
    cities = filtered['City'].unique()
    return [{'label': c, 'value': c} for c in cities], cities[0]

# Update graph based on dropdown and radio selection
@callback(
    Output('main-graph', 'figure'),
    Input('state-dropdown', 'value'),
    Input('city-dropdown', 'value'),
    Input('metric-radio', 'value')
)
def update_graph(state, city, metric):
    filtered_df = df[(df['State'] == state) & (df['City'] == city)]

    if metric in ['rating', 'review_length']:
        fig = px.histogram(filtered_df, x=metric, nbins=20)
        if metric == 'rating':
            fig.update_xaxes(dtick=1, title='Rating')
        elif metric == 'review_length':
            fig.update_xaxes(tickmode='auto', nticks=20, title='Review Length')
    else:
        # manual_sentiment
        fig = px.histogram(
            filtered_df,
            x='manual_sentiment',
            color='manual_sentiment',  # Color by sentiment category
            category_orders={'manual_sentiment': ['Negative', 'Neutral', 'Positive']},
            title='Sentiment Distribution'
        )
        fig.update_layout(showlegend=False)


    fig.update_layout(yaxis_title='Count')
    return fig

# Update word cloud image
@callback(
    Output('wordcloud-img', 'src'),
    Input('state-dropdown', 'value'),
    Input('city-dropdown', 'value')
)
def update_wordcloud(state, city):
    filtered_df = df[(df['State'] == state) & (df['City'] == city)]
    text = " ".join(filtered_df['cleaned_review'].dropna().astype(str))
    if not text.strip():
        text = "No reviews available for this selection."
    encoded_image = generate_wordcloud(text)
    return f'data:image/png;base64,{encoded_image}'

# Run app
if __name__ == '__main__':
    app.run(debug=True)
