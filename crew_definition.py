import os
from crewai import Agent, Crew, Task, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from youtube_transcript_api import YouTubeTranscriptApi  # Corrected import
from langchain.tools import Tool

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Instantiate external tools
search_tool = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper()

# Define tools using CrewAI-compatible @tool decorator

# Define tools properly as LangChain tools
duckduckgo_search = Tool(
    name="DuckDuckGo Search",
    func=lambda query: search_tool.run(query),
    description="Searches the web using DuckDuckGo.",
)

wikipedia_search = Tool(
    name="Wikipedia Search",
    func=lambda query: wikipedia.run(query),
    description="Searches Wikipedia for relevant information.",
)

youtube_transcript_extractor = Tool(
    name="YouTube Transcript Extractor",
    func=lambda video_id: "\n".join(
        entry["text"] for entry in YouTubeTranscriptApi.get_transcript(video_id)
    ) if video_id else "Invalid video ID.",
    description="Extracts transcript from a YouTube video given a video ID.",
)



# Define Agents
researcher_web = Agent(
    role="CrewAI Documentation Expert",
    goal="Extract relevant information from the CrewAI documentation, blog posts, and related websites to provide context and best practices.",
    backstory="Has spent countless hours studying the CrewAI documentation and following the community.",
    tools=[duckduckgo_search, wikipedia_search],  
    verbose=True,
    name="Web Researcher",
)

researcher_video = Agent(
    role="CrewAI Video Tutorial Analyst",
    goal="Watch YouTube tutorials and extract code snippets, explanations, and best practices related to agent creation.",
    backstory="A former video game streamer turned AI enthusiast, now focused on mastering CrewAI through video tutorials.",
    tools=[youtube_transcript_extractor],  
    verbose=True,
    name="Video Researcher",
)

agent_architect = Agent(
    role="CrewAI Agent Designer",
    goal="Create agent definitions (Python code or YAML configurations) that fulfill the user's requirements.",
    backstory="Has extensive experience building CrewAI crews for various applications and understands agent design principles.",
    tools=[duckduckgo_search],  
    verbose=True,
    name="Architect",
)

code_explainer = Agent(
    role="CrewAI Code Explainer and Best Practices Advocate",
    goal="Explain the agent code, highlight important considerations, and offer guidance on integration within a larger CrewAI crew.",
    backstory="A seasoned software engineer with a passion for teaching and sharing best practices.",
    tools=[],  
    verbose=True,
    name="Explainer",
)

# Crew Definition
def create_crew(user_prompt):
    # Define tasks
    task_web_research = Task(
        description=f"Research the CrewAI documentation to understand the functionality of agents, tools, and tasks and any best practices around them, paying special attention to: {user_prompt}",
        agent=researcher_web,
    )

    task_video_research = Task(
        description=f"Watch YouTube tutorials about building agents using CrewAI. Find specific examples of agents with distinct roles, goals, and tools related to: {user_prompt}.",
        agent=researcher_video,
    )

    task_design_agent = Task(
        description=f"Based on the research from the other researchers, create the Python code for an agent that meets the user’s request: {user_prompt}.",
        agent=agent_architect,
    )

    task_explain_code = Task(
        description="Explain the agent code that was created. Provide best practices for using it.",
        agent=code_explainer,
    )

    # Create Crew (added Process.sequential)
    crew = Crew(
        agents=[researcher_web, researcher_video, agent_architect, code_explainer],
        tasks=[task_web_research, task_video_research, task_design_agent, task_explain_code],
        process=Process.sequential,  # Ensure Crew has a process type
        verbose=2,
    )
    return crew

