import gradio as gr
from crew_definition import create_crew  # Import the create_crew function

def run_crew(user_prompt):
    """Runs the CrewAI crew based on the user's prompt."""
    crew = create_crew(user_prompt)  # Create the crew using the imported function

    # Run your crew
    result = crew.kickoff()
    return result

# Gradio Interface
iface = gr.Interface(
    fn=run_crew,
    inputs=gr.Textbox(lines=3, placeholder="Describe the CrewAI agent you want to create..."),
    outputs="text",
    title="CrewAI Agent Creator",
    description="Describe the agent you want to create, and this tool will generate a suggested agent definition and explanation.",
)

iface.launch(share=False) # set share to True if you want a public link
