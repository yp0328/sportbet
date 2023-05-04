### sportbet

# Predicting an Optimal NBA Fantasy Lineup

The primary objective of this project is to create a DraftKings NBA fantasy score predictor. Utilizing historical game data obtained from Basketball-Reference.com and DraftKings, a LSTM model is implemented to predict a player’s total fantasy points on a given day. Using these predictions and historical data, an optimization algorithm is created to generate a lineup of 8 players with the highest expected fantasy points.

## Data
I obtained player box-score statistics data from Kaggle, Basketball-Reference.com, and RotoGuru. 
Preprocessing steps (detailed in final code/full_boxscore_data_final) yielded 4 clean dataframes: filtered_1_df (2022-2023 NBA season),  filtered_2_df (2021-2022 NBA season),  filtered_3_df (2020-2021 NBA season),  and filtered_4_df (2022-2023 NBA season). Each of these dfs had all player statistics and total fantasy points for each player’s historical game performance, with each row representing each game a NBA player has played within a particular season.

## Implementation
First, I constructed a LSTM model (detailed in final code/LSTM_final_model). Using the trained LSTM model, I can now predict the total fantasy points for each player. A function ‘predict_next_game_scores’ is constructed to predict a player’s total fantasy points for their next upcoming game. ‘predict_next_game_scores’  creates a sequence of games for each player based on their latest games to return the predicted fantasy points for each player’s next game. This function ‘predict_next_game_scores’ can be modified to include a cutoff date.

Next, I utilized a dynamic programming-based approach (detailed in final code/lineup_v4_check_final) to solve the problem of selecting an optimal fantasy basketball lineup. The dynamic programming-based approach guarantees the optimal lineup and efficiently solves the lineup optimization problem with memoization. 


## Results
The LSTM model achieves a final test accuracy nearing 85% in predicting players’ total fantasy points. The LSTM models for the rest of NBA seasons analyzed achieve similar results, with decreasing low loss and nearing around 80% accuracy. Each LSTM model took around 4 hours to complete. 

The dynamic programming algorithm generates an optimal lineup that is likely to guarantee a profit for each DFS contest. Optimal lineup fantasy scores, generated by a dynamic programming algorithm, differ in around 30 total fantasy points. Though this number may initially seem significant, as evidenced by DraftKings NBA DFS contest results in Fantasy Cruncher, the minimum total fantasy score vs. the optimal (1st place) total fantasy score generally differs in 40-100 points for each NBA DFS contest. If a lineup reaches a minimum total fantasy score, the user is eligible to profit off this lineup DFS entry. Thus, within the NBA DFS for 5/1/2021, the difference of 30 between our optimal predicted lineup vs. ideal lineup falls well within the range for a minimum total fantasy score to guarantee a profit in this optimal predicted lineup construction. As seen in final code, additional NBA DFS optimal lineups for additional game dates exhibit a similar window between the optimal vs. ideal lineup fantasy scores.

