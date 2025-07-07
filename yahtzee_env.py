import numpy as np
import gymnasium as gym
from gymnasium import spaces

from typing import Optional


#when sampling, use a mask to get actions that are valid in the current state
  #observation space will contain the current dice value, the number of rerolls left, and the scorecard(with 1s representing filled slots)

class YahtzeeEnv(gym.Env):

  def __init__(self, rerolls:int = 2):
      # The dice and scorecard
      self.dice = np.zeros(5, dtype=np.int8)
      self.scorecard = np.full(13, -1, dtype=np.int8) # filling this with -1 avoids the problem of being unable to differentiate between a filled w zero and unfilled bubble.
      # This also solves the yahtzee zero problem. If it is filled with a zero, you are ineligible from future yahtzees. If it is filled with a -1, it is unfilled. If > 0,
      # joker rule applies.
      self.yahtzeeZero = False# if the yahtzee slot was filled with a zero
      self.upperSectionScore = 0
      self.yahtzeeBonus = 0

      #set the number of rerolls
      self.rerolls = rerolls
      self.score = 0
      # Observations are dictionaries with the agent's and the target's location.
      # Each location is encoded as an element of {0, ..., `size`-1}^2
      self.observation_space = gym.spaces.Dict(
          {
              "dice": gym.spaces.Box(low=np.array([1,1,1,1,1]), high=np.array([6,6,6,6,6]), dtype=np.int8),
              "scorecard": gym.spaces.Box(low=-1, high=50, shape=(13,), dtype=np.int8),
              "rerolls": gym.spaces.Discrete(3), #first roll is always done automatically so you either have 2,1,or zero rerolls available.
              "yahtzeeZero" : gym.spaces.Discrete(2),# either 0 or 1 for true or false
              "upperSectionScore" : gym.spaces.Discrete(141),#max possible value for uppersection=140 old thought was 91
          }
      )

      # We have 31 possible reroll combinations and 12 different places to chart our action(technically 13, but yahtzees are taken automatically when they happen.
      #furthermore, an entire game can happen without a yahtzee)
      self.action_space = gym.spaces.Discrete(31+13)
      # Dictionary maps the abstract actions to the directions on the grid
      self._action_to_direction = {
          0: np.array([0, 0, 0, 0, 1]),  # dice to reroll
          1: np.array([0, 0, 0, 1, 0]),  # dice to reroll
          2: np.array([0, 0, 0, 1, 1]),  # dice to reroll
          3: np.array([0, 0, 1, 0, 0]),  # dice to reroll
          4: np.array([0, 0, 1, 0, 1]),  # dice to reroll
          5: np.array([0, 0, 1, 1, 0]),  # dice to reroll
          6: np.array([0, 0, 1, 1, 1]),  # dice to reroll
          7: np.array([0, 1, 0, 0, 0]),  # dice to reroll
          8: np.array([0, 1, 0, 0, 1]),  # dice to reroll
          9: np.array([0, 1, 0, 1, 0]),  # dice to reroll
          10: np.array([0, 1, 0, 1, 1]),  # dice to reroll
          11: np.array([0, 1, 1, 0, 0]),  # dice to reroll
          12: np.array([0, 1, 1, 0, 1]),  # dice to reroll
          13: np.array([0, 1, 1, 1, 0]),  # dice to reroll
          14: np.array([0, 1, 1, 1, 1]),  # dice to reroll
          15: np.array([1, 0, 0, 0, 1]),  # dice to reroll
          16: np.array([1, 0, 0, 1, 0]),  # dice to reroll
          17: np.array([1, 0, 0, 1, 1]),  # dice to reroll
          18: np.array([1, 0, 1, 0, 0]),  # dice to reroll
          19: np.array([1, 0, 1, 0, 1]),  # dice to reroll
          20: np.array([1, 0, 1, 1, 0]),  # dice to reroll
          21: np.array([1, 0, 1, 1, 1]),  # dice to reroll
          22: np.array([1, 1, 0, 0, 0]),  # dice to reroll
          23: np.array([1, 1, 0, 0, 1]),  # dice to reroll
          24: np.array([1, 1, 0, 1, 0]),  # dice to reroll
          25: np.array([1, 1, 0, 1, 1]),  # dice to reroll
          26: np.array([1, 1, 1, 0, 0]),  # dice to reroll
          27: np.array([1, 1, 1, 0, 1]),  # dice to reroll
          28: np.array([1, 1, 1, 1, 0]),  # dice to reroll
          29: np.array([1, 1, 1, 1, 1]),  # dice to reroll
          30: np.array([1, 0, 0, 0, 0]),  # dice to reroll
          31: np.array([0,0,0,0,0,0,0,0,0,0,0,0,1]), # Aces(Ones) (0)
          32: np.array([0,0,0,0,0,0,0,0,0,0,0,1,0]), # Twos (1)
          33: np.array([0,0,0,0,0,0,0,0,0,0,1,0,0]), # Threes (2)
          34: np.array([0,0,0,0,0,0,0,0,0,1,0,0,0]), # Fours (3)
          35: np.array([0,0,0,0,0,0,0,0,1,0,0,0,0]), # Fives (4)
          36: np.array([0,0,0,0,0,0,0,1,0,0,0,0,0]), # Sixes (5)
          37: np.array([0,0,0,0,0,0,1,0,0,0,0,0,0]), # 3 of a Kind (6)
          38: np.array([0,0,0,0,0,1,0,0,0,0,0,0,0]), # 4 of a Kind (7)
          39: np.array([0,0,0,0,1,0,0,0,0,0,0,0,0]), # Full House (8)
          40: np.array([0,0,0,1,0,0,0,0,0,0,0,0,0]), # Small Straight (9)
          41: np.array([0,0,1,0,0,0,0,0,0,0,0,0,0]), # Large Straight (10)
          42: np.array([0,1,0,0,0,0,0,0,0,0,0,0,0]), # Chance(total of all five dice) (11)
          43: np.array([1,0,0,0,0,0,0,0,0,0,0,0,0]), # yahtzee (12)
      }

  def _get_action_mask(self):
    mask = np.zeros(self.action_space.n, dtype=np.int8)

    # Reroll actions (0–30)
    if self.rerolls > 0:
        mask[:31] = 1  # all reroll combinations are allowed

    # Score placement actions (31–43)
    for i in range(13):
        if self.scorecard[i] == -1:
            # Extra yahtzee rule: if this is a bonus yahtzee, force to fill upper if available
            if np.all(self.dice == self.dice[0]) and self.scorecard[12] > 0:
                if i <= 5 and self.scorecard[i] == -1:
                    mask[31 + i] = 1
                # Only allow lower-section actions if no upper slots are left
                elif i > 5 and np.all(self.scorecard[:6] != -1):
                    mask[31 + i] = 1
            else:
                mask[31 + i] = 1

    return mask

  def _get_obs(self):
    return {"dice": self.dice, "scorecard": self.scorecard, "rerolls": self.rerolls, "yahtzeeZero": self.yahtzeeZero, "upperSectionScore" : self.upperSectionScore, "actionMask" : self._get_action_mask()}

  def _get_info(self):
      return {"score" : self.score}

  def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
    super().reset(seed=seed)
    self.score = 0
    self.yahtzeeZero = False
    self.dice = np.random.randint(1, 7, size=5)
    self.scorecard=np.full(13, -1, dtype=np.int8)
    self.upperSectionScore=0
    self.yahtzeeBonus=0
    self.rerolls = 2

    observation = self._get_obs()
    info = self._get_info()

    return observation, info

  def step(self, action):
        reward = 0
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        scoreUpdate = self._action_to_direction[action]

        # logic for transitioning to next state
        if(action < 31): # reroll logic
          if(self.rerolls <= 0):
            reward = reward - 100
            #raise Exception("Error in Step: Tried to take a reroll action while out of rerolls")# might replace with just a super duper negative reward
          else: # need to reroll the correct dice
            for i in range(5):
                if scoreUpdate[i] != 0:
                    self.dice[i] = np.random.randint(1, 7)
            self.rerolls = self.rerolls-1
        else:
          self.rerolls = 2
          pick = action-31

          if(self.scorecard[pick] != -1):#empty
            reward = reward - 100
            #raise Exception("Tried to plot score in taken box on scorecard") # might replace with just a super duper negative reward
          else:
            #check for yahtzee bonus. If there is a yahtzee, also check that the following action is valid
            if(self.scorecard[12] > 0 and np.all(self.dice == self.dice[0])):
              self.yahtzeeBonus += 100
              self.score += 100
              reward += 100
              #if(an upper section action is not taken AND there is an upper section action available):
              if(pick > 5 and np.any(self.scorecard[:6] == -1)):
                reward = reward - 100
                #raise Exception("On an extra yahtzee, failed to take an upper section action when one was available")
              #otherwise scoring is handled below


            #update the scorecard, score and reward accordingly
            ################################################
            scoreIncrement = 0

            #check for first yahtzee
            if(pick == 12):
              if(np.all(self.dice == self.dice[0])):
                scoreIncrement = 50
              else:
                scoreIncrement = 0
                self.yahtzeeZero = True

            #update for the upper section of the scorecard
            elif(pick < 6):
              scoreIncrement = np.count_nonzero(self.dice == pick+1)*(pick+1)
              bonus = 0

              if(self.upperSectionScore < 63 and self.upperSectionScore+scoreIncrement >= 63):
                bonus = 35
              self.upperSectionScore = self.upperSectionScore + scoreIncrement

              reward = reward + bonus
              self.score = self.score + bonus

            #update for the lower section of the scorecard
            elif(pick == 6):
              unique_dice, counts = np.unique(self.dice, return_counts=True)
              if np.any(counts >= 3) or (np.all(self.dice == self.dice[0]) and self.scorecard[12] != -1) : # Joker rule for Yahtzee
                  scoreIncrement = np.sum(self.dice)
            elif(pick == 7):
              unique_dice, counts = np.unique(self.dice, return_counts=True)
              if np.any(counts >= 4) or (np.all(self.dice == self.dice[0]) and self.scorecard[12] != -1): # Joker rule for Yahtzee
                  scoreIncrement = np.sum(self.dice)
            elif(pick == 8):
              unique, counts = np.unique(self.dice, return_counts=True)
              if (len(counts) == 2 and 2 in counts and 3 in counts) or np.all(self.dice == self.dice[0]):
                scoreIncrement = 25
            elif(pick == 9):
              unique_sorted_dice = np.unique(self.dice) # np.unique returns sorted unique values
              is_small_straight = False
              # Check for sequences of 4: 1234, 2345, 3456
              # Convert to string to check for substrings for simplicity: e.g., "1234" in "12345"
              s_dice = "".join(map(str, unique_sorted_dice))
              if "1234" in s_dice or "2345" in s_dice or "3456" in s_dice:
                  is_small_straight = True
              if is_small_straight or (np.all(self.dice == self.dice[0]) and self.scorecard[12] != -1): # Joker rule
                  scoreIncrement = 30
            elif(pick == 10):
              unique_sorted_dice = np.unique(self.dice)
              is_large_straight = False
              if len(unique_sorted_dice) == 5: # Must have 5 unique dice
                  # Check for sequences of 5: 12345, 23456
                  s_dice = "".join(map(str, unique_sorted_dice))
                  if "12345" in s_dice or "23456" in s_dice:
                      is_large_straight = True
              if is_large_straight or (np.all(self.dice == self.dice[0]) and self.scorecard[12] != -1): # Joker rule
                  scoreIncrement = 40
            elif(pick == 11):
              scoreIncrement = np.sum(self.dice)

            if scoreIncrement < 0:
                print(f"WARNING: Negative score detected! pick={pick}, dice={self.dice}, rerolls={self.rerolls}")
                raise Exception("Negative score attempted to add to scorecard")

            reward = reward + scoreIncrement
            self.score = self.score + scoreIncrement
            self.scorecard[pick] = scoreIncrement
            #after scoring action, reroll dice
            self.dice = np.random.randint(1, 7, size=5, dtype=np.int8)

        # An environment is completed if and only if the agent has filled out the entire scorecard
        terminated = np.all(self.scorecard != -1)
        if(terminated):
            reward=reward*1.1+10
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

if __name__ == "__main__":
    env = YahtzeeEnv()
    obs, _ = env.reset()
    print(obs)
