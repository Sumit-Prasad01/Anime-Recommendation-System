from utils.helpers import * 
from config.paths_config import * 
from pipeline.prediction_pipeline import hybrid_recommendation

similar_users = find_similar_user(int(11880),USER_WEIGHTS_PATH, USER2USER_ENCODED, USER2USER_DECODED)
print(similar_users)


user_pref = get_user_preferences(11880,RATING_DF, DF)
print(user_pref)


recc = get_user_recommendations(similar_users, user_pref, DF, SYNOPSIS_DF, RATING_DF)
print(recc)

print(hybrid_recommendation(11880))