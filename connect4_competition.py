# -*- coding: utf-8 -*-
import numpy as np
import time
import pandas as pd


def badai(current_matrix, player_number):
    # this will just return a random column to add a piece to.
    # your ai should return a integer between 0 and 6 (inclusive) for the game to add to the board
    return np.random.randint(7)


#def badai1(current_matrix, player_number):
#    # this will just return a random column to add a piece to.
#    # your ai should return a integer between 0 and 6 (inclusive) for the game to add to the board
#    return np.random.randint(7)
    




#%% Define good ai

def good_ai(m): #, weight): # m = current_matrix
    # Define Intro functions ##################################################
        
    def check_line(line,idx,region):
        player = sum(line==2)
        opponent = sum(line==1)        
        if opponent == 0: # Free of opponent's pieces
            region[idx] = player # Region is open (0) or occupied by (#) of player's pieces
        elif player == 0: # Opponent has pieces, player does not
            region[idx] = -opponent
        else: # Region is blocked (may need to change this)
            region[idx] = -99
        idx += 1
        return idx, region
    
    def check_region(m):
        # Track number of connect fours possible and how close they are to completion
        
        # Need to save these region locations
    
        # Region is an array that describes which pieces are present in each of the
        # sixty-nine different possible connect four configurations.
        region = np.zeros(69, int)
        idx = 0
        
        # Check rows
        for i in range(6):
            for j in range(4):
                line = m[i,j:j+4]
                empty = sum(line==0)
                player = sum(line==2)
                opponent = sum(line==1)
                if opponent == 0: # Free of opponent's pieces
                    region[idx] = player # Region is open (0) or occupied by (#) of player's pieces
                elif player == 0: # Opponent has pieces, player does not
                    region[idx] = -opponent
                else: # Region is blocked (may need to change this)
                    region[idx] = -99
                idx += 1
        
        # Check columns
        for j in range(7):
            for i in range(3):
                line = m[i:i+4,j]
                empty = sum(line==0)
                player = sum(line==2)
                opponent = sum(line==1)        
                if opponent == 0: # Free of opponent's pieces
                    region[idx] = player # Region is open (0) or occupied by (#) of player's pieces
                elif player == 0: # Opponent has pieces, player does not
                    region[idx] = -opponent
                else: # Region is blocked (may need to change this)
                    region[idx] = -99
                idx += 1
        
        # Check diagonals
        # Start from bottom left
        line_diag = {}
        line_diag[0] = np.array([m[2,0], m[3,1], m[4,2], m[5,3]])
        
        line_diag[1] = np.array([m[1,0], m[2,1], m[3,2], m[4,3]])
        line_diag[2] = np.array([m[2,1], m[3,2], m[4,3], m[5,4]])
        
        line_diag[3] = np.array([m[0,0], m[1,1], m[2,2], m[3,3]])
        line_diag[4] = np.array([m[1,1], m[2,2], m[3,3], m[4,4]])
        line_diag[5] = np.array([m[2,2], m[3,3], m[4,4], m[5,5]])
        
        line_diag[6] = np.array([m[0,1], m[1,2], m[2,3], m[3,4]])
        line_diag[7] = np.array([m[1,2], m[2,3], m[3,4], m[4,5]])
        line_diag[8] = np.array([m[2,3], m[3,4], m[4,5], m[5,6]])
        
        line_diag[9] = np.array([m[0,2], m[1,3], m[2,4], m[3,5]])
        line_diag[10] = np.array([m[1,3], m[2,4], m[3,5], m[4,6]])
        
        line_diag[11] = np.array([m[0,3], m[1,4], m[2,5], m[3,6]])
        
    
        # Flip matrix upside down
        m = np.flipud(m) 
        
        # Check remaining diagonals
        line_diag[12] = np.array([m[2,0], m[3,1], m[4,2], m[5,3]])
        
        line_diag[13] = np.array([m[1,0], m[2,1], m[3,2], m[4,3]])
        line_diag[14] = np.array([m[2,1], m[3,2], m[4,3], m[5,4]])
        
        line_diag[15] = np.array([m[0,0], m[1,1], m[2,2], m[3,3]])
        line_diag[16] = np.array([m[1,1], m[2,2], m[3,3], m[4,4]])
        line_diag[17] = np.array([m[2,2], m[3,3], m[4,4], m[5,5]])
        
        line_diag[18] = np.array([m[0,1], m[1,2], m[2,3], m[3,4]])
        line_diag[19] = np.array([m[1,2], m[2,3], m[3,4], m[4,5]])
        line_diag[20] = np.array([m[2,3], m[3,4], m[4,5], m[5,6]])
        
        line_diag[21] = np.array([m[0,2], m[1,3], m[2,4], m[3,5]])
        line_diag[22] = np.array([m[1,3], m[2,4], m[3,5], m[4,6]])
        
        line_diag[23] = np.array([m[0,3], m[1,4], m[2,5], m[3,6]])
        
        # Flip matrix back to normal
        m = np.flipud(m) 
        
        for i in range(len(line_diag)):
            idx, region = check_line(line_diag[i],idx,region)
    
        return region
    
    def valid_moves(m):
        # Determine valid moves
        columns_available = np.zeros(7, int)
        for i in range(7):
            columns_available[i] = np.sum(m[:,i] == 0)
        return columns_available # Shows number of moves remaining
    
    # End Intro functions ####################################################

    # Assumes player 1 goes first, then player 2

    valid_columns = valid_moves(m)
    
    weight = 0.9

    if np.sum(np.ravel(m) != 0) == 0: # i.e. you are going first
        weight = 1.1

        # Always go in the middle center if you are going first
        c = 3 # c = column to place piece in 
        
        # Swap 1's and 2's so 1 = opponent and 2 = player (ai)
        m = 3 - m
        m[m == 3] = 0

    # Code for turns 1 and 2
    elif m[5,3] == 0:
        c = 3 # Go in center if unoccupied
    elif m[5,3] == 1 and m[4,3] == 0:
        c = 3
    elif m[5,3] == 2 and m[4,3] == 1 and m[3,3] == 0:
        c = 3
            
    else:
        ## Code for turn 3+
    
        # Identify winning moves
        
        valid_columns_2 = np.zeros([7,7]) # First one defined above
        
        if np.sum(valid_columns != 0) == 1:
            c = np.where(valid_columns != 0)[0][0]
            return c

        #######################################################################
        mode = 2 # Select 2 or 4 ——— 2 = two moves ahead, 4 = four moves ahead    
        #######################################################################        
        
        if mode == 2:
        
            ######## Region code #######   
            region1 = -999*np.ones((7,69), int) # Store results of my first move
            region2 = -999*np.ones((7,7,69), int) # -999 = invalid column
    #        region4 = np.zeros((7,7,7,7,69))
            for i in np.where(valid_columns)[0]: # Skip full columns (0)
                mm = addpiece(m, 2, i) # Player 2 (me) goes in column i
                valid_columns_2[i,:] = valid_moves(mm)
                region1[i,:] = check_region(mm)
                for j in np.where(valid_columns_2[i,:])[0]:
                    mmm = addpiece(mm, 1, j) # Player 1 (Opponent) goes in column j
                    region2[i,j,:] = check_region(mmm)
    #                for k in range(7):
    #                    mmmm = addpiece(mmm, 2, k) # Player 2 (me) goes in column k
    #        #            region[i][j][k] = check_region(mmmm)
    #                    for l in range(7):
    #                        mmmmm = addpiece(mmmm, 1, l) # Player 1 (opponent) goes in column l
    #                        region4[i,j,k,l,:] = check_region(mmmmm)
    
            ######## Evaluate choice #######
    
            # Identify winning move(s)
            win_column = np.zeros(7)
            for i in range(7):
                win_column[i] = np.sum(region1[i,:] == 4) != 0
            
#            win_column = np.array([0,0,0,1,0,0,0],int) # Test case
            if np.sum(win_column) != 0:
                c = np.where(win_column)[0][0]
                return c
                
#            win_possible = np.sum(region2 == 4) != 0
#            lose_possible = np.sum(region2 == -4) != 0
#            np.where(region2 == -4) # Continue this code!!!!
#
#            if win_possible == True:
#                c = win_column
#            elif lose_possible == True:
#                c = lose_column # ??? Need to block it
#
#            ###### Need to finish this here (above) ######
    
            melt_r2 = np.zeros([49,3+69], int)
            analysis = np.zeros([49,10], int)
            objective = np.zeros(49, int)
            idx = 0
            i = 0
            j = 0
            for i in range(7):
                for j in range(7):
                    melt_r2[idx,0] = i
                    melt_r2[idx,1] = j
                    melt_r2[idx,2:-1] = region2[i,j,:]
                    analysis[idx,0] = sum(region2[i,j,:] == -99) # Region cannot win
                    analysis[idx,1] = sum(region2[i,j,:] == -4) # Opponent wins
                    analysis[idx,2] = sum(region2[i,j,:] == -3) # Opponent 1 away from winning
                    analysis[idx,3] = sum(region2[i,j,:] == -2)        
                    analysis[idx,4] = sum(region2[i,j,:] == -1)        
                    analysis[idx,5] = sum(region2[i,j,:] == 0) # Region is unoccupied
                    analysis[idx,6] = sum(region2[i,j,:] == 1)
                    analysis[idx,7] = sum(region2[i,j,:] == 2)
                    analysis[idx,8] = sum(region2[i,j,:] == 3) # Player 1 away from winning
                    analysis[idx,9] = sum(region2[i,j,:] == 4) # Player wins
            
                    # Experiment with these weightings - may be different depending on if you go first or not (more aggressive pays off if first)
                    w_def_to_off = 1.1 # Weight of defensive play to offensive play
                    w_def_to_off = weight
                    
                    weight1 = np.array([0,-1000,-100,-10,-1,0,1,10,100,1000], int)
                    weight2 = np.r_[np.ones(5)*w_def_to_off, np.ones(5)*1.0]
                    weight_final = weight1*weight2
                    temp = 1 + int(np.round(np.dot(analysis[idx,:], weight_final)))

#                    temp = -1000*analysis[idx,1] + -100*analysis[idx,2] + -10*analysis[idx,3] * -1*analysis[idx,4]
#                    temp += 1*analysis[idx,6] + 10*analysis[idx,7] + 100*analysis[idx,8] + 1000*analysis[idx,9]
#                    temp += 1 # 1 is an indicator to mean NOT FULL
                    
                    if np.sum(analysis[idx,:]) == 0: # If the column is FULL
                        objective[idx] = 0 # 0 = column is FULL, move not possible
                    else:
                        objective[idx] = temp 
                        
                    idx += 1
            
            obj_max = np.zeros(7, int)
            obj_min = np.zeros(7, int)
            obj_max_idx = np.zeros(7, int)
            obj_min_idx = np.zeros(7, int)
            blocked = np.zeros(7)
            for i in range(7):
                j=i*7
                
                # Determine if the whole subset it blocked
                if np.sum(objective[j:j+7] == 0) == 7: # Whole column blocked
                    blocked[i] = 1
                    obj_max[i] = 0
                    obj_min[i] = 0
                    obj_max_idx[i] = -99
                    obj_min_idx[i] = -99
                else:
                    # Compute the max and min objective values excluding individual blocked columns
                    obj_max[i] = np.max(objective[j:j+7][objective[j:j+7] != 0])
                    obj_min[i] = np.min(objective[j:j+7][objective[j:j+7] != 0])
                    obj_max_idx[i] = sp.argmax(objective[j:j+7] == obj_max[i]) + j
                    obj_min_idx[i] = sp.argmax(objective[j:j+7] == obj_min[i]) + j
            
            # Compute the mean of obj_max and obj_min and insert these values for
            # the fully blocked columns, insuring they will never be the absolute
            # max or  min (and thus will never get chosen).
            mean_obj_max = int(np.round(np.mean(obj_max[obj_max != 0]))) # Exclude 0's from mean
            mean_obj_min = int(np.round(np.mean(obj_min[obj_min != 0])))
            for i in range(7):
                if blocked[i] == 1:
                    obj_max[i] = mean_obj_max
                    obj_min[i] = mean_obj_min                   
            
            ##### Still need to check code below #####
            
            max_min = np.max(obj_min)
            max_min_idx = np.where(obj_min == max_min)[0]
            max_min_max = np.max(obj_max[max_min_idx])
            max_min_max_idx = np.where(obj_max[max_min_idx] == max_min_max)[0]
            
            order = np.zeros(7,int) # May not need to have more than 1 move option.
            order[0] = max_min_idx[max_min_max_idx][0]
            len_idx = len(np.where(obj_min == max_min))
            if len_idx > 1:
                for i in range(len_idx):
                    order[i+1] = max_min_idx[max_min_max_idx][i]
                for i in range(len_idx,7):
                    order[i+1:-1] = np.random.randint(7)
            
            c = max_min_idx[max_min_max_idx][0]
        
        else:      # Mode == 4   
        
            #### Region Code for 4 moves ####
            
    #        region2 = np.zeros((7,7,69))
            region4 = np.zeros((7,7,7,7,69))
            for i in range(7): # Look at 7 possible moves
                mm = addpiece(m, 2, i) # Player 2 (me) goes in column i
                for j in range(7):
                    mmm = addpiece(mm, 1, j) # Player 1 (Opponent) goes in column j
    #                region2[i,j,:] = check_region(mmm)
                    for k in range(7):
                        mmmm = addpiece(mmm, 2, k) # Player 2 (me) goes in column k
            #            region[i][j][k] = check_region(mmmm)
                        for l in range(7):
                            mmmmm = addpiece(mmmm, 1, l) # Player 1 (opponent) goes in column l
                            region4[i,j,k,l,:] = check_region(mmmmm)
    
            
            ##### Region for 4 moves ahead #####
    
            melt_r4 = np.zeros([49*49,4+69+1], int)
            analysis = np.zeros([49*49,10], int)
            objective = np.zeros(49*49)
            idx = 0
            i = 0
            j = 0
            k = 0
            l = 0
            for i in range(7):
                for j in range(7):
                    for k in range(7):
                        for l in range(7):
                            melt_r4[idx,0] = i
                            melt_r4[idx,1] = j
                            melt_r4[idx,2] = k
                            melt_r4[idx,3] = l
                            melt_r4[idx,4:-1] = region4[i,j,k,l,:]
                            analysis[idx,0] = sum(region4[i,j,k,l,:] == -99)
                            analysis[idx,1] = sum(region4[i,j,k,l,:] == -4)
                            analysis[idx,2] = sum(region4[i,j,k,l,:] == -3)        
                            analysis[idx,3] = sum(region4[i,j,k,l,:] == -2)        
                            analysis[idx,4] = sum(region4[i,j,k,l,:] == -1)        
                            analysis[idx,5] = sum(region4[i,j,k,l,:] == 0)        
                            analysis[idx,6] = sum(region4[i,j,k,l,:] == 1)        
                            analysis[idx,7] = sum(region4[i,j,k,l,:] == 2)        
                            analysis[idx,8] = sum(region4[i,j,k,l,:] == 3)        
                            analysis[idx,9] = sum(region4[i,j,k,l,:] == 4)        
                    
                            # Experiment with these weightings - may be different depending on if you go first or not (more aggressive pays off if first)
                            temp = -1000*analysis[idx,1] + -100*analysis[idx,2] + -10*analysis[idx,3] * -1*analysis[idx,4]
                            temp += 1*analysis[idx,6] + 10*analysis[idx,7] + 100*analysis[idx,8] + 1000*analysis[idx,9]
                            objective[idx] = temp 
                            idx += 1
            
            obj_max = np.zeros(7**3, int)
            obj_min = np.zeros(7**3, int)
            obj_max_idx = np.zeros(7**3, int)
            obj_min_idx = np.zeros(7**3, int)
            for i in range(7**3):
                j=i*7
                obj_max[i] = np.max(objective[j:j+7**3])
                obj_min[i] = np.min(objective[j:j+7**3])
                obj_max_idx[i] = sp.argmax(objective[j:j+7**3] == obj_max[i]) + j
                obj_min_idx[i] = sp.argmax(objective[j:j+7**3] == obj_min[i]) + j
            
            for i in range(7**2):
                j = i*7
            # Figure out how the slices are divided up
            
            max_min = np.max(obj_min)
            max_min_idx = np.where(obj_min == max_min)[0]
            max_min_max = np.max(obj_max[max_min_idx])
            max_min_max_idx = np.where(obj_max[max_min_idx] == max_min_max)[0]
                    
            order = np.argsort(obj_max[max_min_idx])[::-1]
            
            choice = max_min_idx[order] # Usually contains multiple values. Not always though.
            
            # Check for full columns
            column_full = (sum(m[:,:] == 0) == 0) # True = full column
            
            loop = True
            i = 0
            while loop == True:
                if column_full[i] == False:
                    c = choice[i]
                    loop = False
                else:
                    i += 1        

        # Need to figure out how to get 7 different columns in choice variable,
        # instead of large indices        
        
        if c not in np.where(valid_columns != 0)[0]: # Make sure c is a possible move
            c = random.choice(valid_columns)
            
    return c


#%%


def addpiece(current_matrix, player_number, column):
    # This code will add a piece to the column.

    # Check if column already full, if full do not add a piece
    if current_matrix[0, column] != 0:
        # print('column already full. Bad move by player ', player_number)
        return(current_matrix)

    # Get the occupied spaces
    first = np.argmax(current_matrix != 0, 0)

    new_matrix = copy.copy(current_matrix)
    new_matrix[first[column]-1, column] = player_number
    return new_matrix


def check_winner(matrix):
    # This code will check if there is 4 in a row in any direction.

    # Check vertical win
    counter = np.zeros(7)
    for i in np.arange(5):
        repeat = np.logical_and((matrix[i, :] == matrix[i+1, :]), matrix[i+1, :] != 0)  # Check for repeats
        counter = counter*repeat + repeat  # Reset counter if doesn't repeat else add 1
        if np.any(counter == 3):
            return matrix[i, np.argmax(counter == 3)]

    # Check for a horizontal win
    counter = np.zeros(6)
    for i in np.arange(6):
        repeat = np.logical_and((matrix[:, i] == matrix[:, i+1]), matrix[:, i+1] != 0)  # Check for repeats
        counter = counter*repeat + repeat  # Reset counter if doesn't repeat else add 1
        if np.any(counter == 3):
            return matrix[np.argmax(counter == 3), i]

    # Check the two diagnol wins by appending zeros and rotating matricies.
    full = np.append(matrix, np.zeros((6, 5)), 1)
    for i in np.arange(6):
        full[i, :] = np.roll(full[i, :], i, 0)

    # Check vertical win
    counter = np.zeros(12)
    for i in np.arange(5):
        repeat = np.logical_and((full[i, :] == full[i + 1, :]), full[i + 1, :] != 0)  # Check for repeats
        counter = counter * repeat + repeat  # Reset counter if doesn't repeat else add 1
        if np.any(counter == 3):
            return full[i, np.argmax(counter == 3)]

    full = np.append(matrix, np.zeros((6, 5)), 1)
    for i in np.arange(6):
        full[i, :] = np.roll(full[i, :], -i, 0)

    # Check vertical win
    counter = np.zeros(12)
    for i in np.arange(5):
        repeat = np.logical_and((full[i, :] == full[i + 1, :]), full[i + 1, :] != 0)  # Check for repeats
        counter = counter * repeat + repeat  # Reset counter if doesn't repeat else add 1
        if np.any(counter == 3):
            return full[i, np.argmax(counter == 3)]

    return []


def battle_ais(ai1, ai2):
    # randomly pick first palyer
    firstplayer = np.random.randint(2) + 1
    if firstplayer==1:
        p1 = ai1
        p2 = ai2
    else:
        p1 = ai2
        p2 = ai1

    columns = ['First Player', 'Winner', 'Time: '+ ai1.__name__, 'Time: ' + ai2.__name__]
    df = pd.DataFrame([], columns=columns)

    for _ in range(5):
        # Player 1 goes first
        results = single_match(p1, p2)
        current_round = pd.DataFrame([[p1.__name__, results[0], results[1], results[2]]], index=[_*2+1], columns=columns)
        df = df.append(current_round)
        # Player 2 goes first
        results = single_match(p2, p1)
        current_round = pd.DataFrame([[p2.__name__, results[0], results[2], results[1]]], index=[_*2+2], columns=columns)
        df = df.append(current_round)

    print('Game Results:')
    print(df)

    unique, count = np.unique(df['First Player'], return_counts=True)
    ties = 10 - count.sum()
    keep = np.where(unique != 'None')
    time_ai1 = df['Time: ' + ai1.__name__].sum()
    time_ai2 = df['Time: ' + ai2.__name__].sum()
    df2 = pd.DataFrame(np.asarray([count[keep], [time_ai1, time_ai2]]).T, index=unique[keep], columns=['Wins', 'Time'])
    df2['Ties'] = [ties, ties]
    df2['Losses'] = 10 - df2['Wins'] - df2['Ties']
    df2 = df2[['Wins', 'Ties', 'Losses', 'Time']]

    print('Results Summary:')
    print(df2)

    p1score = 1000*df2['Wins'][ai1.__name__] - time_ai1
    p2score = 1000*df2['Wins'][ai2.__name__] - time_ai2

    if p1score > p2score:
        winner = ai1.__name__
    elif p2score > p1score:
        winner = ai2.__name__
    else:
        winner = 'Tie'

    print('Overall Winner: ', winner)

    return winner








def single_match(ai1, ai2):
    matrix = np.zeros((6, 7), int)
    time1 = 0
    time2 = 0

    winner = []
    player = 1

    while (winner == []) and np.any(matrix == 0):
        if player == 1:
            t0 = time.time()
            chosen = ai1(matrix, player)  # Replace function name with your ai function name.
            t1 = time.time()
            time1 = time1 + t1-t0
            matrix = addpiece(matrix, player, chosen)
            player = 2
        else:
            t0 = time.time()
            chosen = ai2(matrix, player)
            t1 = time.time()
            time2 = time2 + t1-t0
            matrix = addpiece(matrix, player, chosen)
            player = 1

        winner = check_winner(matrix)

    if winner == []:
        winner = 'None'
    elif winner == 1:
        winner = ai1.__name__
    else:
        winner = ai2.__name__

    return winner, time1, time2


if __name__ == '__main__':
#    battle_ais(badai, badai1)
    battle_ais(badai, goodai)
