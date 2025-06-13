import re
from textblob import TextBlob
import ollama


MODEL_NAME = 'qwen3:0.6b'  # Or 'phi3:mini' if you prefer

system_prompt = "You are a helpful and concise assistant. Answer the user's questions directly or follow their simple instructions."
messages = [
        {"role": "system", "content": system_prompt}
       ]

# ====== Setup OpenAI ======

# ====== Toolbox ======
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def extract_keywords(text):
    return list(set(text.split()))

def sentiment_analysis(text):
    return TextBlob(text).sentiment

TOOLBOX = {
    "clean_text": clean_text,
    "extract_keywords": extract_keywords,
    "sentiment_analysis": sentiment_analysis
}

# ====== Flow System ======
class FlowNode:
    def __init__(self, name, function):
        self.name = name
        self.function = function
    
    def run(self, input_data):
        print(f"[{self.name}] Running...")
        result = self.function(input_data)
        print(f"[{self.name}] Output: {result}")
        return result

class Flow:
    def __init__(self):
        self.steps = []
    
    def add_node(self, node: FlowNode):
        self.steps.append(node)
    
    def execute(self, input_data):
        result = input_data
        for node in self.steps:
            result = node.run(result)
        return result

# ====== GPT Planning Agent ======
class GPTPlanningAgent:
    def __init__(self, toolbox):
        self.toolbox = toolbox

    def generate_tool_plan(self, goal: str):
        tool_list = ", ".join(self.toolbox.keys())
        prompt = f"""
                You are a pipeline planning assistant. Your job is to help build a text processing pipeline.
                ignore tools not mentioned in the user tasks.
                Available tools: {tool_list}

                Each tool does:
                - clean_text: lowercases and removes punctuation 
                - extract_keywords: splits text into keywords to determine sentiment
                - sentiment_analysis: returns sentiment polarity/subjectivity

                TOOLS DEPENDENCY:
                 sentiment_analysis depends on extract_keywords which depends on clean_text

                Given this goal: "{goal}"
                Return a Python string of tool names (in order) to use.

                RESPOND WITH THIS MANDATORY REQUIRED FORMAT:<TOOLS>"clean_text", "sentiment_analysis"</TOOLS>
                """ 
    
      
          # Add user message to conversation history
        messages.append({"role": "user", "content": prompt})

        # print(f"\nAssistant ({MODEL_NAME}): ", end="", flush=True)

            
            # Stream the response from Ollama
        response = ollama.chat(
                model=MODEL_NAME,
                messages=messages
            )

        assistant_response=response['message']['content']


            # Add assistant's response to conversation history
        messages.append({"role": "assistant", "content": assistant_response})

        tools = re.search(r'<TOOLS>(.*?)</TOOLS>', assistant_response).group(1)
        tool_list = [tool.strip() for tool in tools.split(',')]
        return tool_list

    def plan_flow(self, goal):
        tools = self.generate_tool_plan(goal)
        print(f"\nAgent plan for goal '{goal}': {tools}")
        flow = Flow()
        for tool_name in tools:
            if tool_name in self.toolbox:
                flow.add_node(FlowNode(tool_name, self.toolbox[tool_name]))
            else:
                print(f"⚠️ Unknown tool: {tool_name}")
        return flow

# ====== Run Example ======
if __name__ == "__main__":
    # user_goal = "Check how positive or negative a user's feedback is"
    user_goal = "Check keywords in  a user's feedback only"
    agent = GPTPlanningAgent(toolbox=TOOLBOX)

    flow = agent.plan_flow(user_goal)

    input_text = "I love how quickly the support team resolved my issue!"
    print("\n=== Executing Flow ===\n")
    result = flow.execute(input_text)

    print("\n=== Final Output ===\n", result)
