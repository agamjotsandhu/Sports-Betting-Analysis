from fastapi import FastAPI
from pydantic import BaseModel
import pickle, json, numpy as np

app = FastAPI()

# Load artefacts at startup
xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
imputer    = pickle.load(open('imputer.pkl', 'rb'))
elo_overall = json.load(open('elo_overall.json'))
elo_surface = json.load(open('elo_surface.json'))
features    = json.load(open('features.json'))

def get_welo(player_id, surface, w=0.5):
    r_o = elo_overall.get(str(player_id), 1500)
    r_s = elo_surface.get(surface, {}).get(str(player_id), 1500)
    return (1 - w) * r_o + w * r_s

class MatchRequest(BaseModel):
    p1_id: str
    p2_id: str
    surface: str  # "Hard", "Clay", "Grass"

@app.post("/predict")
def predict(req: MatchRequest):
    welo_p1 = get_welo(req.p1_id, req.surface)
    welo_p2 = get_welo(req.p2_id, req.surface)
    delta_welo = welo_p1 - welo_p2

    # Build feature vector (zeros for rolling stats you don't have at inference)
    input_vec = np.zeros((1, len(features)))
    welo_idx = features.index('delta_WElo')
    input_vec[0, welo_idx] = delta_welo

    input_imputed = imputer.transform(input_vec)
    prob_p1_wins = xgb_model.predict_proba(input_imputed)[0][1]

    return {"p1_win_probability": round(prob_p1_wins, 4)}