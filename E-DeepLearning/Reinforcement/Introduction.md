#ReinforcementLearning 
## Concept
RL will learn from its experience and over time will be able to identify which actions lead to the best rewards.

The **agent** takes an **action** based on the **environment** **state** and the environment returns the **reward** and the next state. The agent learns from trial and error, initially taking random actions and over time identifying the actions that lead to long-term rewards.




![](Pasted%20image%2020230320211349.png)

The task is to find a function that maps from the state to an action
- How far to push the control sticks to keep that helicopter balanced in the air and fly without crash
| State s | Action a|
| ------ | ------- |
| Give the position of helicopter       | how to move the control sticks        |

specify a rewards function (like training a dog)
let them do a thing, if they did it well, rewards (good!), otherwise bad, punish (bad!)
tell them what to do instead of how to do
positive rewards - flying well +1
negative rewards - flying poorly -1000
then the algorithm to automatically figure out how to choose good actions

Applications
- controlling robots
- factory optimization: rearrange things in the factory to maximize throughout and efficiency
- financial (stock) trading: efficient stock execution 
- playing games: chess, card games
