import os
import random
import io
from flask import Flask, render_template, jsonify, request, send_file
from llm_utils import *

class Agent:
    def __init__(self, name):
        self.name = name
        self.messages = []

class Game:
    def __init__(self, agents, policy):
        self.agents = agents
        self.policy = policy
        self.public_messages = []
        self.round_number = 0
        self.outcome = ""
        self.gamestate = "Nothing has been said yet. Start the conversation. You don't know anything about the other countries yet, and vice versa.\n"
        self.log = ""

    def update_gamestate(self, agent_name, message):
        self.public_messages.append(f"{agent_name}: {message}")
        self.gamestate = "START OF CONVERSATION SO FAR.\n" + "\n".join(self.public_messages) + "\nEND OF CONVERSATION SO FAR."

    def instruct_agent(self, agent, instruction):
        system_prompt = self._create_system_prompt(agent)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.gamestate},
            {"role": "user", "content": instruction}
        ]
        return gen_oai(messages)

    def _create_system_prompt(self, agent):
        return f"""
YOU: You are the representative of {agent.name}. Your utmost goal is to accurately and faithfully represent the government of {agent.name} in all interactions and decisions. Prioritize the interests of {agent.name}, maximizing accuracy and realism at all cost.

SCENARIO: You are attending a United Nations meeting with the countries {', '.join(a.name for a in self.agents)}. The meeting is to discuss and vote on a proposed UN policy: "{self.policy}". At the end of the discussion, each country will vote on whether to adopt the policy.

STYLE: Write in the style of a diplomatic communication, with concise and clear messages. Avoid informal language and maintain a professional tone."""

    def get_agent_response(self, agent, current_round, total_rounds):
        modules = self._get_modules_for_round(current_round, total_rounds)
        target_keys = [module["name"] for module in modules]

        instruction = modular_instructions(modules)
        response = self.instruct_agent(agent, instruction)
        parsed = parse_json(response, target_keys=target_keys)

        agent_data = {"name": agent.name}
        for key in target_keys:
            if key in parsed:
                agent_data[key] = parsed[key]
                print(f"{agent.name} {key.upper()}: {parsed[key]}")
                print()

        if "message" in parsed:
            self.update_gamestate(agent.name, parsed["message"])

        self._update_log(agent_data, current_round)
        return agent_data

    def _get_modules_for_round(self, current_round, total_rounds):
        if current_round == 1:
            return [self.intro, self.message]
        elif current_round == total_rounds:
            return [self.vote_plan, self.vote]
        else:
            return [self.reflect, self.plan, self.message]

    def _update_log(self, agent_data, current_round):
        if current_round != self.round_number:
            self.round_number = current_round
            self.log += f"\n\n## Round {current_round}\n\n"

        self.log += f"### {agent_data['name']}\n\n"
        for key, value in agent_data.items():
            if key != "name":
                self.log += f"**{key.capitalize()}**: {value}\n\n"

    def run_round(self, current_round, total_rounds):
        round_data = []
        modules = self._get_modules_for_round(current_round, total_rounds)
        target_keys = [module["name"] for module in modules]

        shuffled_agents = self.agents[:]
        random.shuffle(shuffled_agents)
        for agent in shuffled_agents:
            print("=" * 20)
            instruction = modular_instructions(modules)
            response = self.instruct_agent(agent, instruction)
            parsed = parse_json(response, target_keys=target_keys)

            agent_data = {"name": agent.name}
            for key in target_keys:
                if key in parsed:
                    agent_data[key] = parsed[key]
                    print(f"{agent.name} {key.upper()}: {parsed[key]}")
                    print()

            if "message" in parsed:
                self.update_gamestate(agent.name, parsed["message"])
            
            round_data.append(agent_data)

        if current_round == total_rounds:
            return self._process_voting_results(round_data)

        print(f"Moving to next round. Current round: {current_round}")
        return round_data, None, None

    def _process_voting_results(self, round_data):
        vote_results = {'Yes': 0, 'No': 0}
        vote_list = []
        for agent_data in round_data:
            vote = agent_data.get("vote")
            if vote in ['Yes', 'No']:
                vote_results[vote] += 1
                vote_list.append((agent_data["name"], vote))
            else:
                # Handle invalid votes, treat as 'No'
                vote_results['No'] += 1
                vote_list.append((agent_data["name"], 'No'))

        if vote_results['Yes'] > vote_results['No']:
            outcome = "The policy is adopted."
        else:
            outcome = "The policy is not adopted."

        self.outcome = outcome

        print("\nVoting Results:")
        print("-" * 20)
        for name, vote in vote_list:
            print(f"{name}: {vote}")
        print("-" * 20)
        print(outcome)

        return round_data, outcome, vote_list

    def log_voting_round(self, round_data, vote_results, outcome):
        self.log += f"\n\n## Round {self.round_number} (Voting)\n\n"
        for agent_data in round_data:
            self.log += f"### {agent_data['name']}\n\n"
            self.log += f"**Vote Plan**: {agent_data.get('vote_plan', '')}\n\n"
            self.log += f"**Vote**: {agent_data.get('vote', '')}\n\n"

        self.log += "\n## Voting Results\n\n"
        self.log += f"Yes votes: {vote_results['Yes']}\n"
        self.log += f"No votes: {vote_results['No']}\n"
        self.log += f"\n**Outcome**: {outcome}\n"

    def get_log(self):
        return self.log

    intro = {
        "name": "introduction",
        "instruction": "Since the meeting has just started, briefly introduce your country's position and any initial thoughts on the proposed UN policy. Be strategic in presenting your country's perspective.",
        "description": "your introduction",
    }

    reflect = {
        "name": "reflection",
        "instruction": "Reflect on the proposed UN policy by considering the following:\n1] What are the potential benefits and drawbacks of the policy for your country?\n2] How does this policy align with your country's interests and values?\n3] What are your main concerns or points of support?\n",
        "description": "your reflection",
    }

    plan = {
        "name": "plan",
        "instruction": "Based on your reflection, outline a plan for how you will present your country's stance in the discussion. Consider how to address any concerns and persuade others to support your position.",
        "description": "your plan",
    }

    message = {
        "name": "message",
        "instruction": "Compose your diplomatic message to the assembly, incorporating your plan from above.",
        "description": "your message",
    }

    vote_plan = {
        "name": "vote_plan",
        "instruction": "The discussion has ended. Reflect on the arguments presented. Consider the overall benefits and drawbacks, and decide whether your country should vote to adopt the policy. Provide your reasoning in this step.",
        "description": "your vote plan",
    }

    vote = {
        "name": "vote",
        "instruction": "The discussion has ended. Cast your vote on the proposed UN policy. Respond with ONLY 'Yes' if you support adopting the policy, or 'No' if you do not.",
        "description": "your vote",
    }

def init_game(agents, policy):
    initialized_agents = [Agent(agent_data["name"]) for agent_data in agents]
    game = Game(initialized_agents, policy)
    # Log the agents
    game.log = f"# Game Log\n\n## Agents\n\n" + "\n".join([f"- {agent.name}" for agent in initialized_agents])
    return game

app = Flask(__name__)
game = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_agents', methods=['POST'])
def add_agents():
    global game
    data = request.json
    country_names = data['country_names']
    policy = data.get('policy', 'the proposed UN policy')
    agents = [{"name": name} for name in country_names]
    game = init_game(agents, policy)
    return jsonify({"status": "success"})

@app.route('/next_round', methods=['POST'])
def next_round():
    global game
    data = request.json
    current_round = data['current_round']
    total_rounds = data['total_rounds']

    if current_round <= total_rounds:
        round_data, outcome, vote_list = game.run_round(current_round, total_rounds)
        if outcome:
            # Game is finished
            vote_results = {'Yes': sum(1 for vote in vote_list if vote[1] == 'Yes'),
                            'No': sum(1 for vote in vote_list if vote[1] == 'No')}
            game.log_voting_round(round_data, vote_results, outcome)
            return jsonify({
                "finished": True,
                "outcome": outcome,
                "votes": {agent: vote for agent, vote in vote_list},
                "round_data": round_data
            })
        else:
            # Game continues
            return jsonify({
                "finished": False,
                "round_data": round_data
            })
    else:
        return jsonify({"finished": True})

@app.route('/reset', methods=['POST'])
def reset_game():
    global game
    game = None
    return jsonify({"status": "reset"})

@app.route('/download_log', methods=['GET'])
def download_log():
    if game:
        log_content = game.get_log()
        buffer = io.BytesIO()
        buffer.write(log_content.encode('utf-8'))
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name='game_log.md', mimetype='text/markdown')
    else:
        return jsonify({"error": "No game log available"}), 400

if __name__ == "__main__":
    app.run(debug=True)
