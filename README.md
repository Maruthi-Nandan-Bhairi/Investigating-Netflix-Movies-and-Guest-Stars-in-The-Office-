# Investigating-Netflix-Movies-and-Guest-Stars-in-The-Office

Investigating The Office: Viewership, Ratings, and Guest Stars
This notebook explores the episode-level data from The Office to understand how audience size and ratings changed over time, which episodes stood out, and whether listed guest stars were associated with different viewership.

# Examine relationships among key numeric episode metrics
metrics = episodes[['Season', 'Ratings', 'Votes', 'Viewership', 'Duration']]

plt.figure(figsize=(8, 5))
sns.heatmap(
    metrics.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    fmt='.2f'
)
plt.title('Correlation among episode metrics')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.figsize'] = (11, 6)

DATA_PATH = Path('the_office_series-selected-columns.csv')
episodes = pd.read_csv(DATA_PATH)
episodes = episodes.drop(columns=[episodes.columns[0]])
episodes['Date'] = pd.to_datetime(episodes['Date'].str.strip(), errors='coerce')
episodes['GuestStars'] = episodes['GuestStars'].fillna('').str.strip()
episodes['has_guest_star'] = episodes['GuestStars'].ne('')
episodes['EpisodeNumber'] = episodes.groupby('Season').cumcount() + 1
episodes.info()
episodes.head()
<class 'pandas.DataFrame'>
RangeIndex: 188 entries, 0 to 187
Data columns (total 11 columns):
 #   Column          Non-Null Count  Dtype         
---  ------          --------------  -----         
 0   Season          188 non-null    int64         
 1   EpisodeTitle    188 non-null    str           
 2   About           188 non-null    str           
 3   Ratings         188 non-null    float64       
 4   Votes           188 non-null    int64         
 5   Viewership      188 non-null    float64       
 6   Duration        188 non-null    int64         
 7   Date            188 non-null    datetime64[us]
 8   GuestStars      188 non-null    str           
 9   has_guest_star  188 non-null    bool          
 10  EpisodeNumber   188 non-null    int64         
dtypes: bool(1), datetime64[us](1), float64(2), int64(4), str(3)
memory usage: 15.0 KB
Season	EpisodeTitle	About	Ratings	Votes	Viewership	Duration	Date	GuestStars	has_guest_star	EpisodeNumber
0	1	Pilot	The premiere episode introduces the boss and s...	7.5	4936	11.2	23	2005-03-24		False	1
1	1	Diversity Day	Michael's off color remark puts a sensitivity ...	8.3	4801	6.0	23	2005-03-29		False	2
2	1	Health Care	Michael leaves Dwight in charge of picking the...	7.8	4024	5.8	22	2005-04-05		False	3
3	1	The Alliance	Just for a laugh, Jim agrees to an alliance wi...	8.1	3915	5.4	23	2005-04-12		False	4
4	1	Basketball	Michael and his staff challenge the warehouse ...	8.4	4294	5.0	23	2005-04-19		False	5
Data quality and season overview
The source contains one row per episode. Blank guest-star fields are treated as episodes without a listed guest star, while numeric columns are kept in their source units.

summary = episodes.groupby('Season').agg(
    episodes=('EpisodeTitle', 'size'),
    avg_viewership=('Viewership', 'mean'),
    avg_rating=('Ratings', 'mean'),
    guest_star_episodes=('has_guest_star', 'sum')
).round(2)
summary
episodes	avg_viewership	avg_rating	guest_star_episodes
Season				
1	6	6.37	7.97	1
2	22	8.17	8.44	6
3	23	8.49	8.59	1
4	14	8.55	8.56	1
5	26	8.76	8.49	3
6	26	7.77	8.20	3
7	24	7.31	8.31	5
8	24	5.39	7.60	3
9	23	4.14	7.91	6
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.lineplot(data=summary.reset_index(), x='Season', y='avg_viewership', marker='o', ax=axes[0], color='#d95f02')
axes[0].set(title='Average viewership by season', ylabel='Average viewership (millions)')
sns.lineplot(data=summary.reset_index(), x='Season', y='avg_rating', marker='o', ax=axes[1], color='#1b9e77')
axes[1].set(title='Average IMDb rating by season', ylabel='Average rating')
plt.tight_layout()

Episode-level trends
Viewership is plotted against the original air date so the trajectory is visible without assuming that seasons had identical lengths.

import numpy as np

fig, ax = plt.subplots(figsize=(14, 6))
sns.scatterplot(data=episodes, x='Date', y='Viewership', hue='Season', palette='tab10', s=65, ax=ax)
date_ordinals = episodes['Date'].map(pd.Timestamp.toordinal).to_numpy()
trend_coefficients = np.polyfit(date_ordinals, episodes['Viewership'], 1)
trend_dates = pd.date_range(episodes['Date'].min(), episodes['Date'].max(), periods=100)
trend_values = np.polyval(trend_coefficients, trend_dates.map(pd.Timestamp.toordinal))
ax.plot(trend_dates, trend_values, color='black', linewidth=2, label='Linear trend')
ax.set(title='The Office episode viewership over time', xlabel='Air date', ylabel='Viewership (millions)')
ax.legend(title='Season', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

top_viewership = episodes.nlargest(10, 'Viewership')[['Season', 'EpisodeTitle', 'Viewership', 'Ratings', 'GuestStars']]
top_ratings = episodes.nlargest(10, 'Ratings')[['Season', 'EpisodeTitle', 'Ratings', 'Viewership', 'GuestStars']]
print('Top episodes by viewership')
display(top_viewership)
print('Top episodes by rating')
display(top_ratings)
Top episodes by viewership
Season	EpisodeTitle	Viewership	Ratings	GuestStars
77	5	Stress Relief	22.91	9.7	Cloris Leachman, Jack Black, Jessica Alba
0	1	Pilot	11.20	7.5	
17	2	The Injury	10.30	9.1	
40	3	The Return	10.20	8.8	
39	3	Traveling Salesmen	10.12	8.6	
41	3	Ben Franklin	10.11	8.1	
60	4	Chair Model	9.81	8.0	
15	2	Christmas Party	9.70	8.9	
51	4	Fun Run	9.70	8.8	
94	6	Niagara: Part 1	9.42	9.4	
Top episodes by rating
Season	EpisodeTitle	Ratings	Viewership	GuestStars
137	7	Goodbye, Michael	9.8	8.42	
187	9	Finale	9.8	5.69	Joan Cusack, Ed Begley Jr, Rachel Harris, Nanc...
77	5	Stress Relief	9.7	22.91	Cloris Leachman, Jack Black, Jessica Alba
59	4	Dinner Party	9.5	9.22	
186	9	A.A.R.M.	9.5	4.56	
27	2	Casino Night	9.4	7.60	
94	6	Niagara: Part 1	9.4	9.42	
95	6	Niagara: Part 2	9.4	9.42	
132	7	Threat Level Midnight	9.4	6.41	
50	3	The Job	9.3	7.88	
Guest-star analysis
The comparison below is descriptive rather than causal. A guest-star credit may coincide with a special episode, a different season, or another production factor, so the result should not be interpreted as proof that guest stars caused a change in audience size.

guest_summary = episodes.groupby('has_guest_star').agg(
    episodes=('EpisodeTitle', 'size'),
    avg_viewership=('Viewership', 'mean'),
    median_viewership=('Viewership', 'median'),
    avg_rating=('Ratings', 'mean')
).round(2)
guest_summary.index = guest_summary.index.map({True: 'Listed guest star', False: 'No listed guest star'})
guest_summary
episodes	avg_viewership	median_viewership	avg_rating
has_guest_star				
No listed guest star	159	7.21	7.51	8.24
Listed guest star	29	7.43	7.60	8.20
guest_credits = episodes.loc[episodes['has_guest_star'], ['GuestStars']].assign(
    GuestStar=lambda frame: frame['GuestStars'].str.split(', ')
).explode('GuestStar')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=episodes, x='has_guest_star', y='Viewership', hue='has_guest_star', legend=False, ax=axes[0], palette=['#7570b3', '#e7298a'])
axes[0].set(title='Viewership by guest-star listing', xlabel='', ylabel='Viewership (millions)')
axes[0].set_xticks([0, 1], ['No listed guest star', 'Listed guest star'])
star_order = guest_credits['GuestStar'].value_counts().index
sns.countplot(data=guest_credits, y='GuestStar', order=star_order, ax=axes[1], color='#66a61e')
axes[1].set(title='Guest-star credit frequency', xlabel='Episodes', ylabel='Guest star')
plt.tight_layout()

1. Executive Summary

This project combines two entertainment-data analytics tasks: investigating Netflix movies and analyzing guest appearances in the television series The Office. The objective is to transform raw entertainment data into meaningful insights using Python and present the results through an interactive Power BI dashboard.

2. Project Objectives

Analyze Netflix movie distribution and trends.

Investigate release years, duration, ratings and genres where available.

Identify changes in Netflix movie content over time.

Analyze guest stars and guest appearances in The Office.

Rank guest stars by appearance frequency.

Compare guest appearances across seasons and episodes.

Create reproducible visualizations and an interactive dashboard.

3. Key Analytical Questions

Netflix Movies

How has the number of Netflix movies changed over time?

Which genres are most common?

What is the typical movie duration?

Which release years contribute the most movies?

How are movies distributed by rating?

Does movie duration vary across genres or ratings?

The Office Guest Stars

Which guest stars appear most frequently?

Which seasons contain the most guest appearances?

Which episodes contain notable guest appearances?

Are appearances concentrated in particular periods?

Which guest stars have repeated appearances?

4. Dataset Description

The project should use the Netflix and The Office CSV files supplied for analysis. Because dataset versions can use different column names, the Python workflow first inspects columns, data types, missing values and duplicates before creating derived variables.

Typical Netflix fields: title, type, release_year, duration, rating, listed_in, country, date_added.

Typical The Office fields: season, episode, episode_title, guest_star, guest_appearance, air_date. These are representative fields; the final code should use the actual CSV schema.

5. Data Cleaning and Preparation

Load CSV files with Pandas.

Inspect shape, columns, data types and missing values.

Remove accidental index columns.

Convert dates and numeric fields to appropriate types.

Handle missing values.

Remove exact duplicates where appropriate.

Normalize text and category labels.

Create derived fields such as decade, duration group, season and appearance count.

6. Exploratory Data Analysis

Netflix analysis examines catalog size, release trends, genres, duration, ratings and other available attributes. The Office analysis counts guest appearances, ranks guest stars, and examines season and episode patterns.

7. Recommended Visualizations

Netflix movies by year — line chart.

Netflix movies by decade — bar chart.

Top genres — horizontal bar chart.

Movie duration distribution — histogram.

Duration by rating — box plot.

Rating distribution — bar chart.

Top The Office guest stars — horizontal bar chart.

Guest appearances by season — column chart.

Guest appearances by episode — bar chart.

Guest-star appearance distribution — histogram or Pareto chart.

8. Statistical and Analytical Methods

Descriptive statistics: count, mean, median, standard deviation, minimum and maximum.

Frequency and percentage analysis.

Group-by aggregation by year, genre, rating, season and guest star.

Correlation analysis for suitable numeric variables.

Outlier detection using box plots and IQR.

9. Power BI Dashboard Layout

Page 1 — Netflix Overview

KPI: Total Movies

KPI: Average Duration

KPI: Latest Release Year

Movies by Year

Movies by Decade

Top 10 Genres

Rating Distribution

Filters: Year, Genre, Rating

Page 2 — Netflix Content Analysis

Duration distribution

Duration by rating

Genre versus average duration

Country contribution if available

Interactive slicers

Page 3 — The Office Guest Stars

Total guest appearances

Unique guest stars

Top 10 guest stars

Guest appearances by season

Guest appearances by episode

Guest-star detail table

Page 4 — Interactive Insights

Guest-star search/filter

Season and year slicers

Genre and rating slicers

Drill-through to detailed records

Tooltips with additional metadata

10. Suggested Power BI DAX

Total Movies = CALCULATE(COUNTROWS(Netflix), Netflix[type] = "Movie")

Average Duration = AVERAGE(Netflix[duration_minutes])

Total Guest Appearances = COUNTROWS(TheOffice)

Unique Guest Stars = DISTINCTCOUNT(TheOffice[guest_star])

Movies by Year = COUNTROWS(Netflix)

11. Project Workflow

Collect and validate datasets.

Load data into Python.

Perform data-quality checks.

Clean and transform data.

Conduct EDA.

Create visualizations.

Export cleaned data and outputs.

Load data into Power BI.

Build interactive dashboards.

Interpret findings and document limitations.

12. Expected Insights

The completed analysis should reveal patterns in Netflix movie availability, genre composition, release timing, duration and ratings, while the The Office analysis should identify guest-star concentration and season or episode patterns. Exact numerical findings should be taken directly from the final CSV analysis outputs.

13. Limitations

Netflix catalogs can vary by country and change over time.

Ratings and metadata may contain missing or inconsistent values.

Genre fields may contain multiple categories in one record.

Guest-star names may have spelling or formatting differences.

Appearance counts do not necessarily measure the importance of a role.

Observational patterns should not automatically be interpreted as causal relationships.

14. Future Enhancements

Add country-level Netflix maps.

Build a recommendation model using movie metadata.

Perform NLP on movie descriptions.

Add IMDb or Rotten Tomatoes information.

Create an episode-level character network for The Office.

Add Power BI drill-through and dynamic tooltips.

15. Skills Demonstrated

Data Cleaning • Exploratory Data Analysis • Data Visualization • Statistical Analysis • Feature Engineering • Python • Pandas • NumPy • Matplotlib • Seaborn • Power BI • DAX • Dashboard Design • Data Storytelling • GitHub Documentation

16. Suggested GitHub Structure

Investigating-Netflix-and-The-Office/
├── data/
│   ├── netflix_movies.csv
│   └── the_office_guest_stars.csv
├── src/
│   └── entertainment_analysis.py
├── notebooks/
│   └── analysis.ipynb
├── outputs/
│   ├── cleaned_data/
│   └── graphs/
├── powerbi/
│   └── dashboard_layout.md
├── docs/
│   └── project_documentation.docx
├── requirements.txt
└── README.md
 
 
