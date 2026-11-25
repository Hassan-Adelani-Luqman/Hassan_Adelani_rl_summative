# Afya Sentinel: video script (5:00)

The live simulation and terminal are on screen from 0:00 to 4:30, which is 4:30 of 5:00. The brief needs at least 3:45 (¾). Only the last 30 seconds are a results slide.

## Before you record (not part of the video)

- Camera on and **share the entire screen**.
- Terminal on the left (about 40% of the screen, large font) and the browser on the right.
- Start the terminal in the folder that contains the repo.
- Run `uv sync` once beforehand so the first launch is quick.
- Have a results slide ready: report Table 8 and Figure 2 (`assets/figures/cumulative_reward.png`).
- The demo is district seed 1000000 with the published DQN model, the same season as report Figure 4, so the moments below happen every time.
- At `--speed 0.25` one season takes about 2 min 24 s. **Space** pauses and resumes. If the season runs ahead of your narration, pause while you talk.

## Script

| Time | On screen | Say |
|---|---|---|
| **0:00–0:30** Problem and launch | Terminal: `ls`, `cd Hassan_Adelani_rl_summative`, `ls`, then `uv run play.py --speed 0.25`. The browser opens on the viewer; press **Space** at once to hold the season at day 0. | "Hi, I'm [name]. Afya Sentinel is a reinforcement-learning agent that decides where to send one malaria screening van around eight villages of Bugesera District during a 30-day rainy season. Its goal is to prevent infections and catch cases early with limited fuel, test kits and bed nets. Each new infection costs one point and a severe case three. Finding a case early earns back at most one, so the agent can never profit from letting malaria spread." |
| **0:30–1:10** Agent View | Still paused at day 0. Point at the pillars, the stock gauges, the **Agent Thinking** panel and the terminal header. Press **Space** to resume: Ntarama, screen, nets, then travel to Rweru (days 0.5–3.5). | "This is the best model, DQN, on a district it never saw in training. Agent View shows only what the agent observes. The pillars are community health worker fever reports, which arrive a day late and miss up to half the cases. Here are fuel, test kits and nets. The Thinking panel shows the value of each of the 12 actions; the highlighted one is its choice, with the reason underneath. The terminal prints every decision, the cases found, the reward and the stock. It starts at Ntarama, the nearest village, screens it, leaves nets and heads for Rweru." |
| **1:10–1:50** Reality View | Press **H**. Point at the glowing prevalence and the red hotspot rings on **Gashora** and **Rweru**. Press **Space** to pause just after the day 7.0 line appears in the terminal (Gashora screen: `19 / 28 / 8`, reward +36.0), then resume. | "H switches to Reality View, the hidden epidemic the agent cannot see. The glow is true prevalence and the red rings are the two hotspots, Gashora and Rweru. Gashora is close to 20% infected, but its reports lag behind. On day 7 the van reaches it: one screening finds 19 early, 28 symptomatic and 8 severe cases, a reward of +36 for one decision." |
| **1:50–2:30** Nets before the rain, then restocking | Days 8.5–10.5: nets at Gashora, then Mayange (point at the rationale: "3-day forecast 63%… 78%"). The van returns to **Nyamata** and the gauges refill. | "Watch what it does before the rain arrives. It places bed nets at Gashora and Mayange when the three-day forecast is 63 and then 78%. Malaria follows rain by about five days, and nets protect a village for ten. Greedy never uses nets; the trained agents spend 16 to 20% of their decisions on them. Now the van is out of kits and nets, so it drives back to Nyamata hospital to restock." |
| **2:30–3:15** Rest of the season | Press **+** twice (about 0.56×). Days 11–30: the van rotates through Mwogo, Juru, Mayange, Ntarama, Rilima and back to Gashora and Rweru. The end-of-season scoreboard appears; also point at the terminal summary. | "I'll speed it up. Through the second half it keeps the same rhythm: screen, leave nets, restock at Nyamata whenever kits or nets run out, and go back to the hotspots when their reports rise. At day 30 the scoreboard compares this district with the baselines: 516 infections with the agent, 1,041 with greedy, which chases the highest reports, and 1,462 doing nothing." |
| **3:15–4:30** Other algorithms and generalisation | Terminal: **Ctrl+C** stops the viewer. Run `uv run play.py --algo ppo --no-browser --episodes 1 --compare-greedy` and scroll to `episode over` and the greedy line. Then run `uv run play.py --scenario heavy_rain`, press **H**, and point at the rain and the flooded roads. | "Ctrl+C stops the viewer. All four algorithms were trained with TorchRL on the same environment, and any of them can be loaded. Here is PPO on the same district, in terminal mode; it is also far ahead of greedy. Finally, generalisation: a heavy-rain scenario the agents never trained on. Roads flood, so trips are slower and use more fuel, and transmission is faster. Across 50 such districts, DQN and PPO still avert over a thousand infections a season, against 256 for greedy." |
| **4:30–5:00** Results slide | Slide with report Table 8 and Figure 2. | "Over 50 held-out districts and three seeds, DQN is best at −427 and the most consistent, PPO is close at −460, and REINFORCE and A2C are around −740 and −790. All four beat greedy's −1,088. Bed nets pay off days later, and DQN's replay buffer suits that delay. One limitation: PPO sometimes runs the van dry on the last day, because the penalty is small; next I would require the van to finish at the hub. Thank you." |

## Timing check

| Segment | Length | Shows live simulation or terminal |
|---|---|---|
| 0:00–0:30 | 0:30 | yes (terminal, launch) |
| 0:30–3:15 | 2:45 | yes (viewer and terminal log) |
| 3:15–4:30 | 1:15 | yes (terminal, PPO and heavy-rain viewer) |
| 4:30–5:00 | 0:30 | no (results slide) |
| **Total** | **5:00** | **4:30 live (90%)** |
