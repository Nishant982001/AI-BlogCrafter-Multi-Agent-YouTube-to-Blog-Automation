__import__('pysqlite3') 
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
from crewai import Agent
from crewai import Crew, Process
from crewai import Task
from crewai_tools import YoutubeChannelSearchTool
from langchain_openai import OpenAI, ChatOpenAI
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

#setting the title
st.set_page_config(page_title="AI BlogCrafter: YouTube-to-Blog Automation ", page_icon="📝")
st.title("AI BlogCrafter: Multi-Agent YouTube-to-Blog Automation")

openai_api_key = st.sidebar.text_input(label="OPENAI API KEY",type="password")

available_models = [
    "gpt-4o-mini",
    "gpt-4o",
]
model=st.sidebar.selectbox("Select an AI Model", available_models)

if not openai_api_key:
    st.info("Please add your nvidia api key to continue")
    st.stop()

llm = ChatOpenAI(model_name=model, openai_api_key=openai_api_key)

channel_name = st.text_input("Enter YouTube Channel ID eg.(@channelname)")
video_title = st.text_input("Enter Video Title for Blog Generation")


blog_reasearcher = Agent(
    role='Blog Reasearcher From Youtube Videos',
    goal = "get the relevent video content for the topic{topic} from the Yt channel",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI data science, machine learning and gen AI and providing suggestions"
    ),
    tools=[],
    llm=llm,
    allow_delegation=True,
)

## create a seniour blog writer agent wiht YT tool

blog_writer = Agent(
    role='Blog Writer',
    goal ="Narrate compelling tech stories about the video{topic} from the Yt channel",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying cmplex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner"
    ),
    tools=[],
    llm=llm,
    allow_delegation=True
)

# Intializze the tool wiht a specific Youtube channel handle to target your search
yt_tool = YoutubeChannelSearchTool(youtube_channel_handle=channel_name)

## Resarch task
research_task = Task(
    description=(
        "Identify the video {topic}."
        "Get detailed information about the video from the channel"
    ),
    expected_output="A comprehensive 3 paragraphs long report based on the {topic} of video content",
    tools=[yt_tool],
    agent=blog_reasearcher
)

## Writing task with language model configuration
write_task = Task(
    description=(
        "get the info from the youtube channel on the topic {topic}."

    ),
    expected_output="Summarize the info from the youtube channel video on the topic{topic} and create the content for the blog",
    tools=[yt_tool],
    agent=blog_writer,
    async_execution=False,
    output_file='new-blog-post.md' ## Example of output customizaiton
)

crew = Crew(
    agents=[blog_reasearcher,blog_writer],
    tasks =[research_task, write_task],
    process = Process.sequential,
    memory = True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

## Start the task execution process with enhanced feedback
if st.button("Generate Blog From Youtube Video"):
    st.info("Processing... Please wait !")
    result=crew.kickoff(inputs={'topic' : video_title})
    st.info("Processing Complete !")
    st.markdown(result)

