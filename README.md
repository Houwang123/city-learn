![Citylearn Banner](https://images.aicrowd.com/uploads/ckeditor/pictures/906/content_Card_Banner.jpg)

# [NeurIPS 2022 Citylearn Challenge](https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge) - Starter Kit 
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/fNRrSvZkry)

This repository is the NeurIPS 2022 Citylearn Challenge **Submission template and Starter kit**! Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent, etc.
*  **Starter code** for you to get started!

# Table of Contents

- [Competition Overview](#competition-overview)
    + [Competition Phases](#competition-phases)
- [Getting Started](#getting-started)
- [How to write your own agent?](#how-to-write-your-own-agent)
- [How to start participating?](#how-to-start-participating)
  * [Setup](#setup)
  * [How do I specify my software runtime / dependencies?](#how-do-i-specify-my-software-runtime-dependencies)
  * [What should my code structure be like?](#what-should-my-code-structure-be-like)
  * [How to make a submission?](#how-to-make-a-submission)
- [Other Concepts](#other-concepts)
    + [Evaluation Metrics](#evaluation-metrics)
    + [Ranking Criteria](#ranking-criteria)
    + [Time constraints](#time-constraints)
  * [Local Evaluation](#local-evaluation)
  * [Contributing](#contributing)
  * [Contributors](#contributors)
- [Important links](#-important-links)


#  Competition Overview
[The CityLearn Challenge 2022](https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge) focuses on the opportunity brought on by home battery storage devices and photovoltaics. It leverages [CityLearn](https://intelligent-environments-lab.github.io/CityLearn/), a Gym Environment for building distributed energy resource management and demand response. The challenge utilizes 1 year of operational electricity demand and PV generation data from 17 single-family buildings in the Sierra Crest home development in Fontana, California, that were studied for _Grid integration of zero net energy communities_.

Participants will develop energy management agent(s) for battery charge and discharge control in each building with the goals of minimizing the net electricity demand from the grid, the monetary cost of electricity drawn from the grid, and the CO<sub>2</sub> emissions when electricity demand is satisfied by the grid.

### Competition Phases
The challenge consists of two phases: 
- In **Phase I**, TODO

- In **Phase II**, TODO

- In **Phase III**, TODO



#  Getting Started
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge).
2. **Install** the Citylearn Simulator [from this link - TODO]().
3. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/citylearn-challenge-2022/citylearn-2022-starter-kit/-/forks/new) to create a fork.
4. **Clone** your forked repo and start developing your autonomous racing agent.
5. **Develop** your controller agents following the template in [how to write your own agent](#how-to-write-your-own-agent) section.
6. [**Submit**](#how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#how-to-make-a-submission). The automated evaluation setup will evaluate the submissions on the citylearn simulator and report the metrics on the leaderboard of the competition.


# How to write your own agent?

We recommend that you place the code for all your agents in the `agents` directory (though it is not mandatory). You should implement the

- `register_reset`
- `compute_action`

**Add your agent name in** `user_agent.py`, this is what will be used for the evaluations.
  
Examples are provided in `agents/random_agent.py` and `agents/rbc_agent.py`.

To make things compatible with [PettingZoo](https://www.pettingzoo.ml/), a reference wrapper is provided that provides observations for each building individually (referred by agent id).

Add your agent code in a way such that the actions returned are conditioned on the `agent_id`. Note that different buildings can have different action spaces. `agents/orderenforcingwrapper.py` contains the actual code that will be called by the evaluator, if you want to bypass it, you will have to match the interfaces, but we recommend using the standard agent interface as shown in the examples.

# How to start participating?

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

2. **Fork the repository**. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/citylearn-challenge-2022/citylearn-2022-starter-kit/-/forks/new) to create a fork.

2.  **Clone the repository**

    ```
    git clone git@gitlab.aicrowd.com:<YOUR_AICROWD_USERNAME>/citylearn-2022-starter-kit.git
    ```

3. **Install** competition specific dependencies!
    ```
    cd citylearn-2022-starter-kit
    pip install -r requirements.txt
    ```

4. Write your own agent as described in [How to write your own agent](#how-to-write-your-own-agent) section.

5. Test your agent locally using `python local_evaluation.py`

6. Make a submission as described in [How to make a submission](#how-to-make-a-submission) section.

## How do I specify my software runtime / dependencies?

We accept submissions with custom runtime, so you don't need to worry about which libraries or framework to pick from.

The configuration files typically include `requirements.txt` (pypi packages), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about the same in the 👉 [runtime.md](docs/runtime.md) file.

## What should my code structure be like?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:

```
.
├── aicrowd.json           # Submission meta information - like your username
├── apt.txt                # Linux packages to be installed inside docker image
├── requirements.txt       # Python packages to be installed
├── local_evaluation.py    # Use this to check your agent evaluation flow locally
├── data/                  # Contains schema files for citylearn simulator (for local testing)
└── agents                 # Place your agents related code here
    ├── random_agent.py            # Random agent
    ├── rbc_agent.py               # Simple rule based agent
    ├── orderenforcingwrapper.py   # Pettingzoo compatibilty wrapper
    └── user_agent.py              # IMPORTANT: Add your agent name here
```

Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!** 

The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "neurips-2022-citylearn-challenge",
  "authors": ["your-aicrowd-username"],
  "description": "(optional) description about your awesome agent",
}
```

This JSON is used to map your submission to the challenge - so please remember to use the correct `challenge_id` as specified above.

## How to make a submission?

👉 [submission.md](/docs/submission.md)

**Best of Luck** :tada: :tada:

# Other Concepts
### Evaluation Metrics
TODO

### Ranking Criteria
TODO

### Time constraints
TODO


## Local Evaluation
- Participants can run the evaluation protocol for their agent locally with or without any constraint posed by the Challenge to benchmark their agents privately. See `local_evaluation.py` for details. You can change it as you like, it will not be used for the competition. You can also change the simulator schema provided under `data/citylearn_challenge_2022_phase_1/schema.json`, this will not be used for the competition.

## Contributing

🙏 You can share your solutions or any other baselines by contributing directly to this repository by opening merge request.

- Add your implemntation as `agents/<your_agent>.py`.
- Test it out using `python local_evaluation.py`.
- Add any documentation for your approach at top of your file.
- Import it in `user_agent.py`
- Create merge request! 🎉🎉🎉 

## Contributors

- [Kingsley Nweye](https://www.aicrowd.com/participants/kingsley_nweye)
- [Zoltan Nagy](https://www.aicrowd.com/participants/nagyz)
- [Dipam Chakraborty](https://www.aicrowd.com/participants/dipam)

# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge

- 🗣 Discussion Forum: https://discourse.aicrowd.com/c/neurips-2022-citylearn-challenge

- 🏆 Leaderboard: https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge/leaderboards
