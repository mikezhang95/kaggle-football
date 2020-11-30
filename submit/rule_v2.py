"""
TODO
   1. 距离球的距离很近的时候，用球的位置而不是下一个位置作为目标
   2. 射门精度问题
   3. 下底传中，直接头球射门
      - @result: 不能头球射门，但是没有延误战机
   4. 进攻的时候增加长传
      - 确实有长传调度了
   

class Action(Enum):
    Idle = 0 Left = 1 TopLeft = 2 Top = 3 TopRight = 4
    Right = 5 BottomRight = 6 Bottom = 7 BottomLeft = 8
    LongPass= 9 HighPass = 10 ShortPass = 11 Shot = 12 
    Sprint = 13 ReleaseDirection = 14 ReleaseSprint = 15 Slide = 16
    Dribble = 17 ReleaseDribble = 18
"""
from kaggle_environments.envs.football.helpers import *
import math
import random
import numpy as np

# side long pass 下底传中
high_pass_state = False   
def high_pass_action(obs, player_x, player_y):
    
    global high_pass_state

    # 球已经传出来了，准备射门
    if high_pass_state and obs["ball_owned_team"] in [0, -1] :  
        return Action.Shot

    # 准备传球
    action_to_release = get_active_sticky_action(obs, ["top", "bottom"])
    if action_to_release != None:
        return action_to_release
    if Action.Top not in obs["sticky_actions"] and Action.Bottom not in obs["sticky_actions"]:
        if player_y > 0:
            return Action.Top
        else:
            return Action.Bottom
    high_pass_state = True
    return Action.HighPass


action_list = [a for a in Action]
def human_readable_obs(obs):
    # Extract observations for the first (and only) player we control.
    obs = obs['players_raw'][0]
    # Turn 'sticky_actions' into a set of active actions (strongly typed).
    obs['sticky_actions'] = { sticky_index_to_action[nr] for nr, action in enumerate(obs['sticky_actions']) if action }
    # Turn 'game_mode' into an enum.
    obs['game_mode'] = GameMode(obs['game_mode'])
    # In case of single agent mode, 'designated' is always equal to 'active'.
    if 'designated' in obs:
        del obs['designated']
    # Conver players' roles to enum.
    obs['left_team_roles'] = [ PlayerRole(role) for role in obs['left_team_roles'] ]
    obs['right_team_roles'] = [ PlayerRole(role) for role in obs['right_team_roles'] ]
    return obs
    
    
def find_patterns(obs, player_x, player_y):
    """ find list of appropriate patterns in groups of memory patterns """
    for get_group in groups_of_memory_patterns:
        group = get_group(obs, player_x, player_y)
        if group["environment_fits"](obs, player_x, player_y):
            return group["get_memory_patterns"](obs, player_x, player_y)

def get_action(obs, player_x, player_y):
    """ get action of appropriate pattern in agent's memory """
    memory_patterns = find_patterns(obs, player_x, player_y)
    # find appropriate pattern in list of memory patterns
    for get_pattern in memory_patterns:
        pattern = get_pattern(obs, player_x, player_y)
        if pattern["environment_fits"](obs, player_x, player_y):
            return pattern["get_action"](obs, player_x, player_y)
        
def get_active_sticky_action(obs, exceptions):
    """ get release action of the first active sticky action, except those in exceptions list """
    release_action = None
    for k in sticky_actions:
        if k not in exceptions and sticky_actions[k] in obs["sticky_actions"]:
            if k == "sprint":
                release_action = Action.ReleaseSprint
            elif k == "dribble":
                release_action = Action.ReleaseDribble
            else:
                release_action = Action.ReleaseDirection
            break
    return release_action

def get_average_distance_to_opponents(obs, player_x, player_y):
    """ get average distance to closest opponents """
    distances_sum = 0
    distances_amount = 0
    for i in range(1, len(obs["right_team"])):
        # if opponent is ahead of player
        if obs["right_team"][i][0] > player_x:
            distance_to_opponent = get_distance(player_x, player_y, obs["right_team"][i][0], obs["right_team"][i][1])
            if distance_to_opponent < 0.13:
                distances_sum += distance_to_opponent
                distances_amount += 1
    # if there is no opponents close around
    if distances_amount == 0:
        return 2, distances_amount
    return distances_sum / distances_amount, distances_amount

def get_far_teamate(obs, player_x, player_y):
    """get farthest teamte so we can make a high pass"""
    
    max_d = [0, 0, 0, 0, 0] # right, topright, bottomright
    
    for i in range(1, len(obs["left_team"])):
        if i == obs["active"]: continue
        if obs["left_team"][i][0] < player_x - 0.1: continue
            
        diff_x = obs["left_team"][i][0] - player_x
        diff_y = (obs["left_team"][i][1] - player_y) * 2.38
        
        # 1. right direction
        if abs(diff_y) < 0.05 * abs(diff_x): 
            distance = get_distance(player_x, player_y, obs["left_team"][i][0], obs["left_team"][i][1])
            if distance > max_d[0]:
                max_d[0] = distance
            continue

        ratio  = abs(diff_y) / (diff_x + 0.001)
        if ratio > 0.9 and ratio < 1.1 :
            distance = get_distance(player_x, player_y, obs["left_team"][i][0], obs["left_team"][i][1])  
            # 4. right up
            if diff_y > 0 and distance > max_d[1]:
                max_d[1] = distance
            # 5. right bottom
            elif diff_y < 0 and distance > max_d[2]:
                max_d[2] = distance
            continue 
                   
        if abs(diff_x) < 0.03 * diff_y: 
            distance = get_distance(player_x, player_y, obs["left_team"][i][0], obs["left_team"][i][1])
            # 2. up direction
            if diff_y > 0 and distance > max_d[3]:
                max_d[3] = distance          
            # 3. bottom direction
            elif diff_y < 0 and distance > max_d[4]:
                max_d[4] = distance            
            continue 
            
    # go directly long pass
    if player_x >= 0.0 :
        thres = [0.5, 0.8, 0.8, 1.1, 1.1]
    else:
        thres = [0.4, 0.4, 0.4, 1.0, 1.0] 
    for i in range(5):
        if max_d[i] < thres[i]:
            max_d[i] = 0.0
    d_id = np.argmax(max_d) 
    direction_set = [Action.Right, Action.TopRight, Action.BottomRight, Action.Top, Action.Bottom]
    if max_d[d_id] == 0:
        if player_x < 0.0:
            obs["high_pass_direction"] = direction_set[0] # the safest direction: right
        else:
            obs["high_pass_direction"] = None
    else:
        obs["high_pass_direction"] = direction_set[d_id]
        
def get_distance(x1, y1, x2, y2):
    """ get two-dimensional Euclidean distance, considering y size of the field """
    return math.sqrt((x1 - x2) ** 2 + (y1 * 2.38 - y2 * 2.38) ** 2)

############# 进攻 ###############

# 下底传中
def bad_angle_high_pass(obs, player_x, player_y):
    """ perform a high pass, if player is at bad angle to opponent's goal """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player have the ball and is at bad angle to opponent's goal
        if high_pass_state or  (obs["ball_owned_player"] == obs["active"] and
                obs["ball_owned_team"] == 0 and
                abs(player_y) > 0.2 and
                player_x > 0.8):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        action = high_pass_action(obs, player_x, player_y)
        return action
    return {"environment_fits": environment_fits, "get_action": get_action}

def close_to_goalkeeper_shot(obs, player_x, player_y):
    """ shot if close to the goalkeeper """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        goalkeeper_x = obs["right_team"][0][0] + obs["right_team_direction"][0][0] * 13
        goalkeeper_y = obs["right_team"][0][1] + obs["right_team_direction"][0][1] * 13
        # player have the ball and located close to the goalkeeper
        if get_distance(player_x, player_y, goalkeeper_x, goalkeeper_y) < 0.3:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if player_y <= -0.05 or (player_y > 0 and player_y < 0.05):
            action_to_release = get_active_sticky_action(obs, ["bottom_right", "sprint"])
            if action_to_release != None:
                return action_to_release
            if Action.BottomRight not in obs["sticky_actions"]:
                return Action.BottomRight
        else:
            action_to_release = get_active_sticky_action(obs, ["top_right", "sprint"])
            if action_to_release != None:
                return action_to_release
            if Action.TopRight not in obs["sticky_actions"]:
                return Action.TopRight
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}

# 解围
def far_from_goal_shot(obs, player_x, player_y):
    """ perform a shot, if far from opponent's goal """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player have the ball and is far from opponent's goal
        if player_x < -0.6 or obs["ball_owned_player"] == 0:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}


# 过人
def go_through_opponents(obs, player_x, player_y):
    """ avoid closest opponents by going around them """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """

        # dribble
        # if right direction is safest
        biggest_distance, final_opponents_amount = get_average_distance_to_opponents(obs, player_x + 0.01, player_y)
        obs["memory_patterns"]["go_around_opponent"] = Action.Right
        
        # if top right direction is safest
        top_right, opponents_amount = get_average_distance_to_opponents(obs, player_x + 0.01, player_y - 0.01)
        if (top_right > biggest_distance and player_y > -0.35): # or (top_right == 2 and player_y > 0.07):
            biggest_distance = top_right
            final_opponents_amount = opponents_amount
            obs["memory_patterns"]["go_around_opponent"] = Action.TopRight
            
        # if bottom right direction is safest
        bottom_right, opponents_amount = get_average_distance_to_opponents(obs, player_x + 0.01, player_y + 0.01)
        if (bottom_right > biggest_distance and player_y < 0.35): # or (bottom_right == 2 and player_y < -0.07):
            biggest_distance = bottom_right
            final_opponents_amount = opponents_amount
            obs["memory_patterns"]["go_around_opponent"] = Action.BottomRight
        
        # is player is surrounded?
        obs["memory_patterns"]["opponent_number"] = final_opponents_amount
        
        # find high pass direction
        get_far_teamate(obs, player_x, player_y)
#         obs["high_pass_direction"] = None
        return True
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """        
        # >=2人，先长传
        if obs["memory_patterns"]["opponent_number"] >= 2:
            if obs["high_pass_direction"] is not None:
                direction_action =  obs["high_pass_direction"]
                if direction_action not in obs["sticky_actions"]:
                    action_to_release = get_active_sticky_action(obs, ["sprint"])
                    if action_to_release != None:
                        return action_to_release
                    return direction_action
            return Action.HighPass
   
        # 0人，1人 带球   
        if obs["memory_patterns"]["go_around_opponent"] not in obs["sticky_actions"]:
            action_to_release = get_active_sticky_action(obs, ["sprint"])
            if action_to_release != None:
                return action_to_release
            return obs["memory_patterns"]["go_around_opponent"]
        if Action.Sprint not in obs["sticky_actions"]:
            return Action.Sprint
        return obs["memory_patterns"]["go_around_opponent"]
    
    return {"environment_fits": environment_fits, "get_action": get_action}


############# 防守 ###############
def run_to_ball_bottom(obs, player_x, player_y):
    """ run to the ball if it is to the bottom from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the bottom from player's position
        if (obs["memory_patterns"]["ball_next_coords"]["y"] > player_y and
                abs(obs["memory_patterns"]["ball_next_coords"]["x"] - player_x) < 0.02):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Sprint not in obs["sticky_actions"]:
            return Action.Sprint
        return Action.Bottom
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_bottom_left(obs, player_x, player_y):
    """ run to the ball if it is to the bottom left from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the bottom left from player's position
        if (obs["memory_patterns"]["ball_next_coords"]["x"] < player_x and
                obs["memory_patterns"]["ball_next_coords"]["y"] > player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Sprint not in obs["sticky_actions"]:
            return Action.Sprint
        return Action.BottomLeft
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_bottom_right(obs, player_x, player_y):
    """ run to the ball if it is to the bottom right from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the bottom right from player's position
        if (obs["memory_patterns"]["ball_next_coords"]["x"] > player_x and
                obs["memory_patterns"]["ball_next_coords"]["y"] > player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Sprint not in obs["sticky_actions"]:
            return Action.Sprint
        return Action.BottomRight
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_left(obs, player_x, player_y):
    """ run to the ball if it is to the left from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the left from player's position
        if (obs["memory_patterns"]["ball_next_coords"]["x"] < player_x and
                abs(obs["memory_patterns"]["ball_next_coords"]["y"] - player_y) < 0.02):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Sprint not in obs["sticky_actions"]:
            return Action.Sprint
        return Action.Left
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_right(obs, player_x, player_y):
    """ run to the ball if it is to the right from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the right from player's position
        if (obs["memory_patterns"]["ball_next_coords"]["x"] > player_x and
                abs(obs["memory_patterns"]["ball_next_coords"]["y"] - player_y) < 0.02):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Sprint not in obs["sticky_actions"]:
            return Action.Sprint
        return Action.Right
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_top(obs, player_x, player_y):
    """ run to the ball if it is to the top from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the top from player's position
        if (obs["memory_patterns"]["ball_next_coords"]["y"] < player_y and
                abs(obs["memory_patterns"]["ball_next_coords"]["x"] - player_x) < 0.02):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Sprint not in obs["sticky_actions"]:
            return Action.Sprint
        return Action.Top
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_top_left(obs, player_x, player_y):
    """ run to the ball if it is to the top left from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the top left from player's position
        if (obs["memory_patterns"]["ball_next_coords"]["x"] < player_x and
                obs["memory_patterns"]["ball_next_coords"]["y"] < player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Sprint not in obs["sticky_actions"]:
            return Action.Sprint
        return Action.TopLeft
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_ball_top_right(obs, player_x, player_y):
    """ run to the ball if it is to the top right from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the top right from player's position
        if (obs["memory_patterns"]["ball_next_coords"]["x"] > player_x and
                obs["memory_patterns"]["ball_next_coords"]["y"] < player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Sprint not in obs["sticky_actions"]:
            return Action.Sprint
        return Action.TopRight  
    return {"environment_fits": environment_fits, "get_action": get_action}        
     
 
def idle(obs, player_x, player_y):
    """ do nothing, stickly actions are not affected (player maintains his directional movement etc.) """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        return True
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        action_to_release = get_active_sticky_action(obs, [])
        if action_to_release != None:
            return action_to_release
        return Action.Idle
    
    return {"environment_fits": environment_fits, "get_action": get_action}

############# 特殊的场景，角球、点球等 ###############
def corner(obs, player_x, player_y):
    """ perform a high pass in corner game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is corner game mode
        if high_pass_state or obs['game_mode'] == GameMode.Corner:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        action = high_pass_action(obs, player_x, player_y)
        return action
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def free_kick(obs, player_x, player_y):
    """ perform a high pass or a shot in free kick game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is free kick game mode
        if obs['game_mode'] == GameMode.FreeKick:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        # shot if player close to goal
        if player_x > 0.5:
            action_to_release = get_active_sticky_action(obs, ["top_right", "bottom_right"])
            if action_to_release != None:
                return action_to_release
            if Action.TopRight not in obs["sticky_actions"] and Action.BottomRight not in obs["sticky_actions"]:
                if player_y > 0:
                    return Action.TopRight
                else:
                    return Action.BottomRight
            return Action.Shot
        # high pass if player far from goal
        else:
            action_to_release = get_active_sticky_action(obs, ["right"])
            if action_to_release != None:
                return action_to_release
            if Action.Right not in obs["sticky_actions"]:
                return Action.Right
            return Action.HighPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def goal_kick(obs, player_x, player_y):
    """ perform a short pass in goal kick game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is goal kick game mode
        if obs['game_mode'] == GameMode.GoalKick:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        action_to_release = get_active_sticky_action(obs, ["top_right", "bottom_right"])
        if action_to_release != None:
            return action_to_release
        # randomly choose direction
        if Action.TopRight not in obs["sticky_actions"] and Action.BottomRight not in obs["sticky_actions"]:
            if random.random() < 0.5:
                return Action.TopRight
            else:
                return Action.BottomRight
        return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def kick_off(obs, player_x, player_y):
    """ perform a short pass in kick off game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is kick off game mode
        if obs['game_mode'] == GameMode.KickOff:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        action_to_release = get_active_sticky_action(obs, ["top", "bottom"])
        if action_to_release != None:
            return action_to_release
        if Action.Top not in obs["sticky_actions"] and Action.Bottom not in obs["sticky_actions"]:
            if player_y > 0:
                return Action.Top
            else:
                return Action.Bottom
        return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def penalty(obs, player_x, player_y):
    """ perform a shot in penalty game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is penalty game mode
        if obs['game_mode'] == GameMode.Penalty:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        action_to_release = get_active_sticky_action(obs, ["top_right", "bottom_right"])
        if action_to_release != None:
            return action_to_release
        # randomly choose direction
        if Action.TopRight not in obs["sticky_actions"] and Action.BottomRight not in obs["sticky_actions"]:
            if random.random() < 0.5:
                return Action.TopRight
            else:
                return Action.BottomRight
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def throw_in(obs, player_x, player_y):
    """ perform a short pass in throw in game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is throw in game mode
        if obs['game_mode'] == GameMode.ThrowIn:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        action_to_release = get_active_sticky_action(obs, ["right"])
        if action_to_release != None:
            return action_to_release
        if Action.Right not in obs["sticky_actions"]:
            return Action.Right
        return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}


############# 找patterns ###############

# 进攻
def offence_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for environments in which player's team has the ball """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player ownes the ball or very close to the ball when opponent doesnt have the ball
        distance_to_ball = get_distance(player_x, player_y, obs["ball"][0], obs["ball"][1])
        if (obs["ball_owned_team"] == 0) or (obs["ball_owned_team"] != 1 and ( high_pass_state or distance_to_ball < 0.03) ): 
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            bad_angle_high_pass, # 下底传中
            close_to_goalkeeper_shot, # 射门
            far_from_goal_shot, # 解围
            go_through_opponents, # 过人
            idle
        ]
        return memory_patterns
 
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

# 防守
def defence_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for environments in which opponent's team has the ball """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player don't have the ball
        if obs["ball_owned_team"] != 0 :
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        
#         # Notice: 为了更加保守
#         # shift ball x position, if opponent has the ball
#         if obs["ball_owned_team"] == 1:
#             obs["memory_patterns"]["ball_next_coords"]["x"] -= 0.05   

        if abs(player_x - obs["ball"][0]) < 0.05 and abs(player_y - obs["ball"][1]) < 0.05:
            obs["memory_patterns"]["ball_next_coords"] = {
                    "x": obs["ball"][0] + obs["ball_direction"][0] * 1,
                    "y": obs["ball"][1] + obs["ball_direction"][1] * 0.2
           }
        
        memory_patterns = [
            run_to_ball_right,
            run_to_ball_left,
            run_to_ball_bottom,
            run_to_ball_top,
            run_to_ball_top_right,
            run_to_ball_top_left,
            run_to_ball_bottom_right,
            run_to_ball_bottom_left,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

# 特殊
def special_game_modes_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for special game mode environments """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # if game mode is not normal
        if obs['game_mode'] != GameMode.Normal:
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            corner,
            free_kick,
            goal_kick,
            kick_off,
            penalty,
            throw_in,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

# 门将解围
def goalkeeper_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for goalkeeper """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player is a goalkeeper have the ball
        if (obs["ball_owned_player"] == obs["active"] and
                obs["ball_owned_team"] == 0 and
                obs["ball_owned_player"] == 0):
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            far_from_goal_shot,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}


# 兜底
def other_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for all other environments """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        return True
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}



############# 按照角色分类 ###############
############# 特殊/守门员/进攻/防守/其他 ###############
# list of groups of memory patterns
groups_of_memory_patterns = [
    special_game_modes_memory_patterns,
    goalkeeper_memory_patterns,
    offence_memory_patterns,
    defence_memory_patterns,
    other_memory_patterns
]

# dictionary of sticky actions
sticky_actions = {
    "left": Action.Left,
    "top_left": Action.TopLeft,
    "top": Action.Top,
    "top_right": Action.TopRight,
    "right": Action.Right,
    "bottom_right": Action.BottomRight,
    "bottom": Action.Bottom,
    "bottom_left": Action.BottomLeft,
    "sprint": Action.Sprint,
    "dribble": Action.Dribble
}

############ main function ############

# @human_readable_agent wrapper modifies raw observations 
# provided by the environment:
# https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#raw-observations
# into a form easier to work with by humans.
# Following modifications are applied:
# - Action, PlayerRole and GameMode enums are introduced.
# - 'sticky_actions' are turned into a set of active actions (Action enum)
#    see usage example below.
# - 'game_mode' is turned into GameMode enum.
# - 'designated' field is removed, as it always equals to 'active'
#    when a single player is controlled on the team.
# - 'left_team_roles'/'right_team_roles' are turned into PlayerRole enums.
# - Action enum is to be returned by the agent function.
# @human_readable_agent

def agent(raw_obs):
    """ Ole ole ole ole """
    # rule observation
    obs = human_readable_obs(raw_obs)
    # raw observation
    obs["raw_obs"] = raw_obs

    # 球已经被抢断了
    global high_pass_state
    if high_pass_state and obs["ball_owned_team"] == 1 : high_pass_state = False
    # shift positions of opponent team players
    for i in range(len(obs["right_team"])):
        obs["right_team"][i][0] += obs["right_team_direction"][i][0]
        obs["right_team"][i][1] += obs["right_team_direction"][i][1]
    # dictionary for Memory Patterns data
    obs["memory_patterns"] = {}
    # coordinates of the ball in the next step
    obs["memory_patterns"]["ball_next_coords"] = {
        "x": obs["ball"][0] + obs["ball_direction"][0] * 10,
        "y": obs["ball"][1] + obs["ball_direction"][1] * 2
    }
    # We always control left team (observations and actions
    # are mirrored appropriately by the environment).
    controlled_player_pos = obs["left_team"][obs["active"]]
    # get action of appropriate pattern in agent's memory
    action = get_action(obs, controlled_player_pos[0], controlled_player_pos[1])
    # return action
    return [action.value]

    