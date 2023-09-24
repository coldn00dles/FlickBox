import numpy as np
import pandas as pd 
import ast
from sentence_transformers import SentenceTransformer, util
movies = pd.read_csv('input/tmdb_5000_movies.csv')
movies[['genres','keywords','production_companies','production_countries','spoken_languages']] = movies[['genres','keywords','production_companies','production_countries','spoken_languages']].applymap(lambda x : ast.literal_eval(str(x)))
def get_data(x, cols, dict):
    for col in cols: 
        for i in range(len(x[col])):
            for j in range(len(x[col][i])):
                x[col][i][j] = x[col][i][j][dict]
    return x

movies = get_data(movies, ['genres','keywords','production_companies','production_countries','spoken_languages'],'name')
movies[['budget','id','popularity','revenue','runtime','vote_average','vote_count']] = movies[['budget','id','popularity','revenue','runtime','vote_average','vote_count']].apply(pd.to_numeric, errors = 'coerce')

credits = pd.read_csv("input/tmdb_5000_credits.csv")

credits[['cast', 'crew']] = credits[['cast', 'crew']].applymap(lambda x : ast.literal_eval(str(x)))

credits = get_data(credits, ['cast', 'crew'],'name')

movies = pd.merge(movies, credits[['movie_id','cast', 'crew']],  left_on= "id", right_on = "movie_id", how = "left")

movies['overview'] = movies['overview'].astype(str)

model = SentenceTransformer('all-MiniLM-L6-v2')
overview_embeddings = model.encode(movies['overview'])

overview_cos_sim = util.cos_sim(overview_embeddings, overview_embeddings)

def recommender(movie_name,num_movies):
    result = pd.concat([movies["original_title"], 
                    pd.DataFrame(overview_cos_sim[:,movies[movies["original_title"] == movie_name].index].numpy(), columns=['Overview'])],axis = 1)
    result = result[result["Overview"] != 1]
    result = result.sort_values('Overview', ascending= False).head(num_movies + 1).reset_index(drop =  True)
    result = result['original_title']
    res = result.tolist()
    return "\n".join(res[1:])

import gradio as gr
demo = gr.Interface(
    fn=recommender,
    inputs=["text", gr.Slider(0, 15,step=1)],
    outputs=["text"]
)
demo.launch()