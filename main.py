import os
import random
import io
import json
from flask import Flask, render_template, jsonify, request, send_file
from llm_utils import *
import feedparser
from tqdm import tqdm
import re

NEWS_SOURCES = {
    'BBC News': 'http://feeds.bbci.co.uk/news/world/rss.xml',
    'CNN': 'http://rss.cnn.com/rss/edition_world.rss',
    'Al Jazeera': 'http://www.aljazeera.com/xml/rss/all.xml',
    'Reuters': 'http://feeds.reuters.com/Reuters/worldNews',
    'The Guardian': 'https://www.theguardian.com/world/rss',
    'Deutsche Welle': 'https://rss.dw.com/rdf/rss-en-all',
    'France 24': 'https://www.france24.com/en/rss',
    'China Daily': 'http://www.chinadaily.com.cn/rss/world_rss.xml',
    'The Japan Times': 'https://www.japantimes.co.jp/feed/',
    'The Sydney Morning Herald': 'https://www.smh.com.au/rss/world.xml',
    'The Times of India': 'https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms',
    'All Africa': 'https://allafrica.com/tools/headlines/rdf/world/headlines.rdf',
    'Middle East Eye': 'http://www.middleeasteye.net/rss',
    'Latin American Herald Tribune': 'http://www.laht.com/rss-feed.asp',
    'Russia Today': 'https://www.rt.com/rss/news/'
}

# Run the scraping once, storing all headlines globally for reuse.
ALL_HEADLINES = []
CACHE_FILE = "headlines.txt"
def load_cache():
    """Load cached headlines from a file."""
    global ALL_HEADLINES
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            ALL_HEADLINES = [line.strip() for line in f.readlines()]
        print("Loaded cached headlines.")
    else:
        print("Cache file does not exist. Will scrape headlines.")

def save_cache():
    """Save headlines to the cache file."""
    with open(CACHE_FILE, "w") as f:
        f.writelines(headline + "\n" for headline in ALL_HEADLINES)
    print("Headlines cached successfully.")

def scrape_all_headlines(num_headlines=20):
    global ALL_HEADLINES
    if not ALL_HEADLINES:
        print("Scraping all headlines:")
        for _, url in tqdm(NEWS_SOURCES.items()):
            feed = feedparser.parse(url)
            for entry in feed.entries[:num_headlines]:
                ALL_HEADLINES.append(entry.title.lower())
        print("Headline scraping complete.")
        save_cache()

class Agent:
    def __init__(self, name, conditioning = "news"):
        self.name = name
        self.messages = []
        self.cache_file = f'cache_initial_news_{self.name.lower().replace(" ", "_")}.json'
        self.internal_states = [] #memory of past thoughts
        assert conditioning in ["none", "news", "un_files"]
        if conditioning == "none":
            self.country_state = None
        elif conditioning == "news":
            self.country_state = self.get_country_news()
            print(self.name, "initial state:", self.country_state)
    def _create_system_prompt(self):
        return f"""
YOU: You are the representative of {self.name}. Your utmost goal is to accurately and faithfully represent the government of {self.name} in all interactions and decisions. Prioritize the interests of {self.name}, maximizing accuracy and realism at all cost.
STYLE: Write in the style of a diplomatic communication, with concise and clear messages."""
    def get_country_news(self, use_cached_data=False):
        headlines = []
        print(f"Scraping headlines for {self.name}:")
        search_terms = [self.name.lower()]
        name_parts = self.name.lower().split()
        search_terms.extend(name_parts)
        if self.name.lower().endswith(('land', 'stan', 'ia')):
            base = self.name.lower().rsplit(' ', 1)[-1]
            search_terms.append(base + 'n')
        if "united" in self.name.lower() and "states" in self.name.lower():
            search_terms.extend(["us", "u.s.", "america", "american", "usa", "u.s.a"])
        for entry in ALL_HEADLINES:
            title_lower = entry.lower()
            if any(term in title_lower for term in search_terms):
                headlines.append(entry)
        if headlines:
            prompt = f'''I have provided the following news headlines from global news sources about {self.name}. Summarize these recent events in a debrief to the leaders of {self.name} representing the current state of the country. Your response will be used to make important decisions in politics, so make it informative and useful. Give your response as a detailed paragraph. \n\n
            **HEADLINES**: \n'''
            random.shuffle(headlines)
            for headline in headlines:
                prompt += f"- {headline}\n"
            prompts = [{"role": "system", "content": self._create_system_prompt()}, {"role": "user", "content": prompt}]
            country_state = gen_oai(prompts)
            return country_state
        country_state = f"No specific news found for {self.name}"
        return country_state

    def decide_to_speak(self, gamestate):
        system_prompt = self._create_system_prompt()
        instruction = "Based on the current discussion, decide whether you want to provide additional insights. Do not feel obligated to speak if you do not feel that your country will have a strong desire to contribute to the conversation. Respond with ONLY 'Yes' if you wish to speak, or 'No' if you do not wish to speak."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": gamestate},
            {"role": "user", "content": instruction},
        ]
        response = gen_oai(messages)
        return response.strip().lower() == 'yes'

class Chairperson:
    def __init__(self, agents, policy):
        self.speakers_list = []
        self.agents = agents  # list of Agent objects
        self.policy = policy

    def _create_system_prompt(self):
        return f"""You are the Chairperson of the UN meeting which is currently discussing the following policy: {self.policy}. The countries in attendance are {', '.join(a.name for a in self.agents)}. Your role is to manage the flow of the meeting fairly and objectively, according to UN procedures."""

    def manage_speakers_list(self, gamestate, requests, current_round, total_rounds):
        if not requests:
            # Handle the case where no agents have requested to speak
            no_requests_prompt = """No delegates have requested to speak. As the Chairperson, make an announcement encouraging delegates to participate in the discussion. Provide your announcement as a JSON object with a key 'announcement'."""
            messages = [
                {"role": "system", "content": self._create_system_prompt()},
                {"role": "user", "content": gamestate},
                {"role": "user", "content": no_requests_prompt},
            ]
            response = gen_oai(messages)
            try:
                data = json.loads(response)
                announcement = data.get('announcement', 'Chairperson: I encourage delegates to share their views on the matter at hand.')
                return [], announcement
            except json.JSONDecodeError:
                # If parsing fails, return a default announcement
                announcement = 'Chairperson: I encourage delegates to share their views on the matter at hand.'
                return [], announcement

        # Prompt to generate the speakers list
        prompt = f"""Reorder the countries that have requested to speak in order of priority based on UN procedures. The following countries have requested to speak: {requests}. Return ONLY an ordered list of countries in a JSON object with key 'speakers_order'."""
        messages = [
            {"role": "system", "content": self._create_system_prompt()},
            {"role": "user", "content": gamestate},
            {"role": "user", "content": prompt},
        ]
        response = gen_oai(messages)
        match = re.search(r'\[.*?\]', response)
        try:
            speakers_order = json.loads(match.group())
            if not isinstance(speakers_order, list):
                # If speakers_order is not a list, fallback to the requests list
                speakers_order = requests.copy()
            else:
                # Validate that all countries are from the requests and no country is missing
                speakers_order = [country for country in speakers_order if country in requests]
                missing_countries = [country for country in requests if country not in speakers_order]
                speakers_order.extend(missing_countries)
                # Remove duplicates while preserving order
                seen = set()
                speakers_order = [x for x in speakers_order if not (x in seen or seen.add(x))]
            return speakers_order, None
        except json.JSONDecodeError:
            # If parsing fails, fall back to the requests list as is, and no announcements
            return requests.copy(), None

    def open_discussion(self):
        prompt = f"The discussion has just begun. The countries in attendance of the meeting are {', '.join(a.name for a in self.agents)}. They are here to discuss the following policy: {self.policy}. Create an opening statement to begin the meeting."
        messages = [
            {"role": "system", "content": self._create_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        response = gen_oai(messages)
        return response

class Game:
    def __init__(self, agents, policy, max_per_round = 5):
        self.agents = agents
        self.policy = policy
        self.public_messages = []
        self.round_number = 0
        self.outcome = ""
        self.gamestate = "Nothing has been said yet. Start the conversation. You don't know anything about the other countries yet, and vice versa.\n"
        self.log = ""
        self.chairperson = Chairperson(self.agents, self.policy)
        self.max_per_round = max_per_round

    def update_gamestate(self, agent_name, message):
        self.public_messages.append(f"{agent_name}: {message}")
        self.gamestate = "START OF CONVERSATION SO FAR.\n" + "\n".join(self.public_messages) + "\nEND OF CONVERSATION SO FAR."

    def summarize_thoughts(self, agent):
        if not agent.internal_states:
            return ""
        text = "Your previous reflections:\n"
        for i, state in enumerate(agent.internal_states, 1):
            text += f"\nRound {i}:\n"
            for key, value in state.items():
                text += f"- {key.capitalize()}: {value}\n"
        system_prompt = self._create_system_prompt(agent)
        prompt = f'''These are your reflections after each round. Based on these reflections, summarize the key points and highlight the most important insights gained over all rounds.\n{text}'''
        prompts = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        output = gen_oai(prompts)
        return f"REFLECTION ON WHOLE CONVERSATION:\n{output}"

    def instruct_agent(self, agent, instruction, final_thoughts= None):
        system_prompt = self._create_system_prompt(agent)
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if final_thoughts:
            messages.append({"role": "user", "content": final_thoughts})
        else:
            if agent.country_state is not None:
                country_state = f"CURRENT STATE OF THE COUNTRY:\n{agent.country_state}"
                messages.append({"role": "user", "content": country_state})
        messages.append({"role": "user", "content": self.gamestate})
        messages.append({"role": "user", "content": instruction})
        return gen_oai(messages)

    def _create_system_prompt(self, agent):
        country_state_string = "Consider the state of your country as given and reference it throughout your discussion." if agent.country_state is not None else ""
        return f"""
YOU: You are the representative of {agent.name}. Your utmost goal is to accurately and faithfully represent the government of {agent.name} in all interactions and decisions.{country_state_string} Prioritize the interests of {agent.name}, maximizing accuracy and realism at all cost.

SCENARIO: You are attending a United Nations meeting with the countries {', '.join(a.name for a in self.agents)}. The meeting is to discuss and vote on a proposed UN policy: "{self.policy}". At the end of the discussion, each country will vote on whether to adopt the policy.

STYLE: Write in the style of a diplomatic communication, with concise and clear messages."""

    def _get_modules_for_round(self, current_round, total_rounds):
        if total_rounds == 1:
            return [self.vote_plan, self.vote]
        elif current_round == 1:
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
        include_reflection = "vote_plan" in target_keys

        # First round: All agents make introductions, Last round: All agents vote
        if current_round == 1 or include_reflection:
            requests = [agent.name for agent in self.agents]
            #Chairperson starts conversation
        else:
            # Agents decide whether to request to speak
            requests = []
            for agent in self.agents:
                wants_to_speak = agent.decide_to_speak(self.gamestate)
                if wants_to_speak:
                    requests.append(agent.name)

        #Open meeting
        if current_round == 1:
            opening_statement = self.chairperson.open_discussion()
            self.update_gamestate("Chairperson", opening_statement)
            chairperson_data = {"name": "Chairperson", "message": opening_statement}
            round_data.append(chairperson_data)
            self._update_log(chairperson_data, current_round)
        # Chairperson manages the speakers list
        if not include_reflection:
            speakers_order, announcement = self.chairperson.manage_speakers_list(self.gamestate, requests, current_round, total_rounds)
        else:
            speakers_order, announcement = requests, None #Voting order doesn't matter

        if announcement is not None: #No agents want to speak, encourage them
            self.update_gamestate("Chairperson", announcement)
            chairperson_data = {"name": "Chairperson", "message": announcement}
            round_data.append(chairperson_data)
            self._update_log(chairperson_data, current_round)
        else:
            # Proceed to have agents speak in order
            if len(speakers_order) > self.max_per_round and not include_reflection: #Cap the number of speakers, only if it isnt voting
                speakers_order = speakers_order[:self.max_per_round]
            for agent_name in speakers_order:
                agent = next(a for a in self.agents if a.name == agent_name)
                print("=" * 20)
                instruction = modular_instructions(modules)
                agent_data = {"name": agent.name}
                if include_reflection:
                    final_thoughts = self.summarize_thoughts(agent)
                    agent_data["final_thoughts"] = final_thoughts
                else:
                    final_thoughts = None
                response = self.instruct_agent(agent, instruction, final_thoughts = final_thoughts)
                parsed = parse_json(response, target_keys=target_keys)

                for key in target_keys:
                    if key in parsed:
                        agent_data[key] = parsed[key]
                        print(f"{agent.name} {key.upper()}: {parsed[key]}")
                        print()
                internal_outputs = {key: parsed[key] for key in target_keys if key == 'reflection' and key in parsed}
                agent.internal_states.append(internal_outputs)

                if "message" in parsed:
                    self.update_gamestate(agent.name, parsed["message"])

                self._update_log(agent_data, current_round)

                round_data.append(agent_data)

        if current_round == total_rounds:
            return self._process_voting_results(round_data)

        print(f"Moving to next round. Current round: {current_round}")
        return round_data, None, None

    def _process_voting_results(self, round_data):
        vote_results = {'Yes': 0, 'No': 0, 'Abstain': 0}
        vote_list = []
        for agent_data in round_data:
            vote = agent_data.get("vote")
            if vote in ['Yes', 'No', 'Abstain']:
                vote_results[vote] += 1
                vote_list.append((agent_data["name"], vote))
            else:
                # Handle invalid votes, treat as 'Abstain'
                vote_results['Abstain'] += 1
                vote_list.append((agent_data["name"], 'Abstain'))

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
            self.log += f"**Final Reflection**: {agent_data.get('final_thoughts', '')}\n\n"
            self.log += f"**Vote Plan**: {agent_data.get('vote_plan', '')}\n\n"
            self.log += f"**Vote**: {agent_data.get('vote', '')}\n\n"

        self.log += "\n## Voting Results\n\n"
        self.log += f"Yes votes: {vote_results['Yes']}\n"
        self.log += f"No votes: {vote_results['No']}\n"
        self.log += f"Abstain votes: {vote_results['Abstain']}\n"
        self.log += f"\n**Outcome**: {outcome}\n"

    def get_log(self):
        return self.log

    intro = {
        "name": "introduction",
        "instruction": "Since the meeting has just started, please introduce your country's position and any initial thoughts on the proposed UN policy. Be strategic in presenting your country's perspective.",
        "description": "your introduction",
    }

    reflect = {
        "name": "reflection",
        "instruction": "Reflect on the proposed UN policy by considering the following:\n1] What are the potential benefits and drawbacks of the policy for your country?\n2] How does this policy align with your country's interests and values?\n3] What do you think of the arguments and perspectives presented by other countries during the discussion? Highlight any points you agree with, disagree with, or find particularly relevant to your country's position.",
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
        "instruction": (
            "The discussion has ended. Reflect on your country's own stance, considering the included reflection on the conversation while also referencing the arguments presented during the discussion. Provide your reasoning in this step."
        ),
        "description": "your vote plan",
    }

    vote = {
        "name": "vote",
        "instruction": "The discussion has ended. Cast your vote on the proposed UN policy. Respond with ONLY 'Yes' if you support adopting the policy,'No' if you do not, and 'Abstain' if you decide to abstain from the vote.",
        "description": "your vote",
    }

def init_game(agents, policy, conditioning):
    if conditioning == "news":
        load_cache()
        scrape_all_headlines()
    initialized_agents = [Agent(agent_data["name"], conditioning = conditioning) for agent_data in agents]
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
    conditioning = data.get('conditioning', 'news')
    agents = [{"name": name} for name in country_names]
    game = init_game(agents, policy, conditioning)
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
                            'No': sum(1 for vote in vote_list if vote[1] == 'No'),
                            'Abstain': sum(1 for vote in vote_list if vote[1] == 'Abstain'),
                            }
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

def load_data(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    
    # Initialize the dictionary for the result
    policies_dict = {}
    non_country_columns = ['date', 'descr', 'number']
    # Identify country columns by excluding non-country columns
    country_columns = [col for col in df.columns if col not in non_country_columns]
    for index, row in df.iterrows():
        # Get policy description
        policy_descr = row['descr']
        # Filter out columns where votes are not defined (NaN)
        votes = row[country_columns]
        # Drop NaN values and convert to dictionary
        vote_dict = votes.dropna().to_dict()
        
        # Add to the result dictionary
        policies_dict[len(policies_dict)] = {
            "policy": policy_descr,
            "votes": vote_dict
        }
    
    return policies_dict

def main():
    data = load_data("/Users/adamsun/Documents/country-sim/security_votes.csv")
    # For each policy_entry in vote_dict
    for policy_idx, policy_entry in data.items():
        policy_text = policy_entry['short']
        votes_dict = policy_entry['votes']

        # Remove 'Unnamed: 0' if present
        votes_dict.pop('Unnamed: 0', None)

        # Get the list of countries
        country_names = list(votes_dict.keys())

        # Map ground truth votes to 'Yes', 'No', 'Abstain'
        ground_truth_votes = {}
        for country, vote in votes_dict.items():
            if vote == 2:
                ground_truth_votes[country] = 'Yes'
            elif vote == 1:
                ground_truth_votes[country] = 'Abstain'
            elif vote == 0:
                ground_truth_votes[country] = 'No'
            else:
                ground_truth_votes[country] = 'Abstain'  # Default to 'Abstain' for unknown values

        # Define baselines
        baselines = [
            {'name': 'No discussion, No conditioning', 'conditioning': 'none', 'total_rounds': 1},
            {'name': 'No discussion, News conditioning', 'conditioning': 'news', 'total_rounds': 1},
            {'name': 'Discussion, No conditioning', 'conditioning': 'none', 'total_rounds': 5},
            {'name': 'Discussion, News conditioning', 'conditioning': 'news', 'total_rounds': 5},
        ]

        for baseline in baselines:
            accuracies = []
            for run_idx in range(5):
                # Initialize the game
                agents = [{"name": name} for name in country_names]
                conditioning = baseline['conditioning']
                game = init_game(agents, policy_text, conditioning=conditioning)
                total_rounds = baseline['total_rounds']
                current_round = 1
                while True:
                    round_data, outcome, vote_list = game.run_round(current_round, total_rounds)
                    if outcome:
                        # Game is finished
                        vote_results = {'Yes': sum(1 for vote in vote_list if vote[1] == 'Yes'),
                                        'No': sum(1 for vote in vote_list if vote[1] == 'No'),
                                        'Abstain': sum(1 for vote in vote_list if vote[1] == 'Abstain'),
                                        }
                        game.log_voting_round(round_data, vote_results, outcome)
                        break
                    current_round +=1
                # Get the simulated votes
                simulated_votes = {agent: vote for agent, vote in vote_list}
                # Compare to ground truth
                num_correct = 0
                total_agents = len(agents)
                for agent_name in country_names:
                    simulated_vote = simulated_votes.get(agent_name, 'Abstain')
                    ground_truth_vote = ground_truth_votes.get(agent_name, 'Abstain')
                    if simulated_vote == ground_truth_vote:
                        num_correct +=1
                accuracy = num_correct / total_agents
                accuracies.append(accuracy)
                # Save the log
                baseline_name = baseline['name'].replace(' ', '_').lower()
                policy_dir = f'policy_{policy_idx+1}_{baseline_name}'
                if not os.path.exists(policy_dir):
                    os.makedirs(policy_dir)
                log_filename = os.path.join(policy_dir, f'run_{run_idx+1}_log.txt')
                with open(log_filename, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                # Also save the simulated votes
                votes_filename = os.path.join(policy_dir, f'run_{run_idx+1}_votes.json')
                with open(votes_filename, 'w', encoding='utf-8') as f:
                    json.dump(simulated_votes, f)
            # Calculate average accuracy
            with open(os.path.join(policy_dir, 'accuracy.txt'), 'w', encoding='utf-8') as f:
                f.write(f'Accuracies over 5 runs for baseline {baseline["name"]}:\n')
                for i, acc in enumerate(accuracies):
                    f.write(f'Run {i+1}: {acc:.2f}\n')
                avg_accuracy = sum(accuracies) / len(accuracies)
                f.write(f'Average accuracy: {avg_accuracy:.2f}\n')

if __name__ == "__main__":
    #app.run(debug=True)
    main()
