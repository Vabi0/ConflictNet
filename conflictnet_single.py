#program for an agentic AI correspodance
import boto3
import json
import time
import sys
from colorama import init, Fore

init(autoreset=True)

bedrock = boto3.client('bedrock', region_name='us-east-1')

# ----------------- Base Agent -----------------
class BaseAgent:
    def __init__(self, name, role, color):
        self.name = name
        self.role = role
        self.color = color

    def type_out(self, text, delay=0.02):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print("\n")

    def call_model(self, model_id, prompt):
        """
        Calls a Nova model via Bedrock invoke_model_with_response_stream and returns concatenated output.
        """
        response = bedrock.invoke_model_with_response_stream(
            modelId=model_id,
            contentType="application/json",
            body=json.dumps({"inputText": prompt})
        )

        output_text = ""
        for event in response['ResponseStream']:
            if 'Payload' in event:
                output_text += event['Payload'].decode('utf-8')
        return output_text


class EfficiencyAgent(BaseAgent):
    def __init__(self):
        super().__init__("Efficiency Agent", "The Pathfinder", Fore.GREEN)

    def think(self, question):
        prompt = "Explain why efficiency should be prioritized in AI decision-making: {}".format(question)
        reasoning = self.call_model("amazon_nova_2_lite", prompt)
        return reasoning

class FairnessAgent(BaseAgent):
    def __init__(self):
        super().__init__("Fairness Agent", "The Guardian", Fore.CYAN)

    def think(self, question):
        prompt = "Explain why fairness should be prioritized in AI decision-making: {}".format(question)
        reasoning = self.call_model("amazon_nova_2_sonic", prompt)
        return reasoning

class CriticAgent(BaseAgent):
    def __init__(self):
        super().__init__("Critic Agent", "The Challenger", Fore.MAGENTA)

    def think(self, efficiency_text, fairness_text):
        combined = "Efficiency: {} | Fairness: {}".format(efficiency_text, fairness_text)
        reasoning = self.call_model("amazon_nova_embedding_multimodal", combined)
        return reasoning

class ModeratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("Moderator Agent", "The Navigator", Fore.YELLOW)

    def think(self, efficiency_text, fairness_text, critic_text):
        prompt = "Synthesize the following debate and give a final recommendation:\nEfficiency: {}\nFairness: {}\nCritic: {}".format(
            efficiency_text, fairness_text, critic_text
        )
        reasoning = self.call_model("amazon_nova_act", prompt)
        return reasoning


class ConflictNet:
    def __init__(self):
        self.efficiency_agent = EfficiencyAgent()
        self.fairness_agent = FairnessAgent()
        self.critic_agent = CriticAgent()
        self.moderator_agent = ModeratorAgent()

    def run_debate(self, question):
        print("\n=== ConflictNet Debate Started ===\n")
        print("Question: {}\n".format(question))
        time.sleep(1)

        # Stage 1: Efficiency Agent
        print(self.efficiency_agent.color + "[Stage 1] Efficiency Agent speaks")
        efficiency_text = self.efficiency_agent.think(question)
        self.efficiency_agent.type_out(efficiency_text)
        time.sleep(1)

        # Stage 2: Fairness Agent
        print(self.fairness_agent.color + "[Stage 2] Fairness Agent speaks")
        fairness_text = self.fairness_agent.think(question)
        self.fairness_agent.type_out(fairness_text)
        time.sleep(1)

        # Stage 3: Critic Agent
        print(self.critic_agent.color + "[Stage 3] Critic Agent interjects")
        critic_text = self.critic_agent.think(efficiency_text, fairness_text)
        self.critic_agent.type_out(critic_text)
        time.sleep(1)

        # Stage 4: Moderator Agent
        print(self.moderator_agent.color + "[Stage 4] Moderator Final Decision")
        moderator_text = self.moderator_agent.think(efficiency_text, fairness_text, critic_text)
        self.moderator_agent.type_out(moderator_text)
        time.sleep(1)

        # Optional: save debate to log file
        with open("debate_log.txt", "a") as log_file:
            log_file.write("Question: {}\n".format(question))
            log_file.write("Efficiency: {}\n".format(efficiency_text))
            log_file.write("Fairness: {}\n".format(fairness_text))
            log_file.write("Critic: {}\n".format(critic_text))
            log_file.write("Moderator: {}\n\n".format(moderator_text))

        print("=== Debate Complete ===\n")


if __name__ == "__main__":
    system = ConflictNet()
    question = input("Enter your AI ethics question:\n> ")
    system.run_debate(question)