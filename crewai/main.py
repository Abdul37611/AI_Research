from crewai import Agent, Task, Process, Crew
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import requests
from bs4 import BeautifulSoup
import uuid
from pathlib import Path
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

app = FastAPI(
    title="CrewAI",
    description="API documentation to interact with the agents.",
)


def fetch_youtube_transcript(video_url: str) -> Optional[str]:
    """
    Fetches the transcript of a YouTube video.

    Given a URL of a YouTube video, this function uses the youtube_transcript_api
    library to fetch the transcript of the video.

    Args:
        video_url (str): The URL of the youTube video.

    Returns:
        Optional[str]: The transcript of the video, or None if any error occurs.
    """
    try:
        # Extract the video ID from the URL
        video_id = video_url.split("v=")[1].split("&")[0]
        
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join(entry["text"] for entry in transcript_list)

        return transcript
    except NoTranscriptFound:
        # Return None if any request-related exception is caught
        return None


def generate_and_save_images(query: str, image_size: str = "1024x1024") -> List[str]:
    """
    Function to paint, draw or illustrate images based on the users query or request. Generates images from a given query using OpenAI's DALL-E model and saves them to disk.  Use the code below anytime there is a request to create an image.

    :param query: A natural language description of the image to be generated.
    :param image_size: The size of the image to be generated. (default is "1024x1024")
    :return: A list of filenames for the saved images.
    """

    client = OpenAI()  # Initialize the OpenAI client
    response = client.images.generate(model="dall-e-3", prompt=query, n=1, size=image_size)  # Generate images

    # List to store the file names of saved images
    saved_files = []

    # Check if the response is successful
    if response.data:
        for image_data in response.data:
            # Generate a random UUID as the file name
            file_name = str(uuid.uuid4()) + ".png"  # Assuming the image is a PNG
            file_path = Path(file_name)

            img_url = image_data.url
            img_response = requests.get(img_url)
            if img_response.status_code == 200:
                # Write the binary content to a file
                with open(file_path, "wb") as img_file:
                    img_file.write(img_response.content)
                    print(f"Image saved to {file_path}")
                    saved_files.append(str(file_path))
            else:
                print(f"Failed to download the image from {img_url}")
    else:
        print("No image data found in the response!")

    # Return the list of saved files
    return saved_files


def fetch_user_profile(url: str) -> Optional[str]:
    """
    Fetches the text content from a personal website.

    Given a URL of a person's personal website, this function scrapes
    the content of the page and returns the text found within the <body>.

    Args:
        url (str): The URL of the person's personal website.

    Returns:
        Optional[str]: The text content of the website's body, or None if any error occurs.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Check for successful access to the webpage
        if response.status_code == 200:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            # Extract the content of the <body> tag
            body_content = soup.find("body")
            # Return all the text in the body tag, stripping leading/trailing whitespaces
            return " ".join(body_content.stripped_strings) if body_content else None
        else:
            # Return None if the status code isn't 200 (success)
            return None
    except requests.RequestException:
        # Return None if any request-related exception is caught
        return None


researcher = Agent(
    role="Researcher",
    goal="Research new AI insights",
    backstory="You are an AI research assistant, you provide assistance in research, providing them with insights",
    verbose=True, 
    allow_delegation=True,
    )

general = Agent(
    role="General Assistant",
    goal="Provide assistance in general tasks",
    backstory="You are a general assistant, capable of handling a variety of tasks",
    verbose=True, 
    allow_delegation=True,
    tools=[fetch_user_profile,generate_and_save_images],
    )

youtube = Agent(
    role="Youtube Assistant",
    goal="Summarize or explain a youtube video",
    backstory="You are a youtube assistant, that get the transcript and then assists the user in understanding the video",
    verbose=True, 
    allow_delegation=True,
    tools=[fetch_youtube_transcript],
    )

class TextData(BaseModel):
  task: str
  model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task": "What is the difference between a convolutional neural network and a recurrent neural network?",
                }
            ]
        }
    }

@app.post("/agents", tags=["CrewAI Agents"])
async def process_text(request: TextData):
  """
  An AI researcher agent that will assist you in your research.
  """
  try:
    task = Task(description = request.task, agent = researcher)

    crew = Crew(
        agents=[researcher,general,youtube],
        tasks=[task],
        verbose=2,
        process=Process.sequential
    )

    result = crew.kickoff()

    return {"data": result}
  except Exception as e:
    return {"error": str(e)}

