# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(pick_up, drop) for pick_up in range(m) for drop in range(m) if pick_up!=drop or drop==0] #all combinations of actions along with (0,0)
        self.state_space = [(location, time, day) for location in range(m) for time in range(t) for day in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input
    #
    # def state_encod_arch1(self, state):
    #     """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
    #
    #     return state_encod


    # Use this function if you are using architecture-2
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        vector_size = m + t + d + m + m
        state_encod = [0]*vector_size

        location_index = state[0] - 1
        state_encod[location_index] = 1

        hour_index = m + state[1]
        state_encod[hour_index] = 1

        day_index = m + t + state[2]
        state_encod[day_index] = 1

        pick_up_index = m + t + d + action[0] - 1
        state_encod[pick_up_index] = 1

        drop_index = m + t + d + m + action[1] - 1
        state_encod[drop_index] = 1

        return state_encod


    ## Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        actions.append([0,0])

        return possible_actions_index,actions

    def calc_new_time_and_day(time_of_day, day_of_week, travel_time):
        """Return new time of day and day of the week after adding the travel time."""
        time_of_day = (time_of_day + travel_time) % (t-1)
        day_of_week = (day_of_week + (travel_time // (t-1)))%(d-1)

        return time_of_day, day_of_week

    def calc_travel_time(current_loc, start_loc, end_loc, time_of_day, day_of_week, Time_matrix):
        """Returns t1: time taken from current location to start locations
                   t2: time taken from start location to end location"""
        if start_loc == 0 and end_loc == 0:
            return 0, 1 #ride not accepeted by driver so increment time by 1 hour

        if current_loc == start_loc:
            t1 = 0
        else:
            t1 = int(Time_matrix[current_loc-1][start_location-1][time_of_day][day_of_week])
            time_of_day, day_of_week = calc_new_time_and_day(time_of_day, day_of_week, t1)

        t2 = int(Time_matrix[start_location-1][end_location-1][time_of_day][day_of_week])

        return t1, t2

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        current_loc = state[0]
        time_of_day = state[1]
        day_of_week = state[2]
        start_loc = action[0]
        end_loc = action[1]

        t1, t2 = calc_travel_time(current_loc, start_loc, end_loc, time_of_day, day_of_week, Time_matrix)

        if start_loc != 0 or end_loc != 0: #action not (0,0)
            reward = R * t2 - C * (t1 + t2)
        else:
            reward = -C

        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        current_loc = state[0]
        time_of_day = state[1]
        day_of_week = state[2]
        start_loc = action[0]
        end_loc = action[1]

        t1, t2 = calc_travel_time(current_loc, start_loc, end_loc, time_of_day, day_of_week, Time_matrix)
        total_travel_time = t1 + t2

        if start_loc != 0 or end_loc != 0:
            new_loc = end_loc
        new_time_of_day, new_day_of_week = calc_new_time_and_day(time_of_day, day_of_week, total_travel_time)

        next_state = (new_loc, new_time_of_day, new_day_of_week)

        return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init
