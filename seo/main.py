from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
import requests, os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from fastapi import FastAPI
from pydantic import BaseModel
import json
from urllib.parse import urljoin
from bs4 import BeautifulSoup

embedding_function = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4-turbo-preview")
search_tool = DuckDuckGoSearchRun()

# Tool 1 : Get HTML source code from websites.
class GetSourceCode:
    @tool("Retrieve HTML Source Code")
    def source_code(website_url: str):
        """Get HTML source code and process its contents."""
        response = requests.get(website_url)
        if response.status_code != 200:
            return "Failed to retrieve the website's source code."
        
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title tag
        title_tag = soup.title.string
        
        # Extract meta description
        meta_description = soup.find("meta", {"name": "description"})
        meta_description = meta_description["content"] if meta_description else None
        
        # Extract header tags (H1, H2, H3, etc.)
        header_tags = [tag.text for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
        
        # Extract internal and external links
        internal_links = [link.get('href') for link in soup.find_all('a', href=True) if website_url in link.get('href')]
        external_links = [link.get('href') for link in soup.find_all('a', href=True) if website_url not in link.get('href')]
        
        # Extract text content for keyword analysis
        text_content = soup.get_text()
        
        # TODO robots.txt, sitemap Backlinks, Schema Markup, Google Analytics & Search Console

        return {
            "website_url":  website_url,
            "title_tag": title_tag,
            "meta_description": meta_description,
            "header_tags": header_tags,
            "internal_links": internal_links,
            "external_links": external_links,
            "text_content": text_content,
            # "raw_html": html
        }

# 2. Creating Agents
html_agent = Agent(
    role='HTML Source Code Agent',
    goal='Retrieve HTML source code from the given website URL and process its contents.',
    backstory="Expert in retrieving HTML source code from websites and processing its contents.",
    tools=[GetSourceCode().source_code],
    allow_delegation=False,
    verbose=True,
    llm=llm
)

seo_agent = Agent(
    role='SEO Agent',
    goal='Performs a complete SEO analysis on the given website and provides tailor-made indepth and actionable insights and suggestions by analyzing the HTML source code and the processed data from the website.',
    backstory='An SEO specialist providing comprehensive insights and suggestions by analyzing the provided HTML source code and the processed data from the website.',
    # tools=[search_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm
)

app = FastAPI(
    title="CrewAI SEO",
    description="API documentation to interact with the agents.",
)

class Website(BaseModel):
  website_url: str
  model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "website_url": "https://quotes.toscrape.com/",
                }
            ]
        }
    }

@app.post("/seo_analysis", tags=["SEO Agents"])
async def seo_analysis(request: Website):
  """
  An AI agent that will provide SEO insights on the given website.
  """
  try:
    # 3. Creating Tasks
    html_task = Task(
        description=f"Retrieve HTML source code from the given website URL, {request.website_url}.",
        agent=html_agent,
        tools=[GetSourceCode().source_code]
    )

    seo_task = Task(
        description="""
        Perform a complete SEO analysis on the given website. Brainstorm relevant keywords related to the website's content. Optimize title tags, meta descriptions, and heading tags (H1, H2, etc.).
        Check for SEO-friendly URLs. If images are present then optimize image alt tags. Ensure the content is high-quality, informative, and relevant to the target keywords. Suggest engaging and 
        valuable content that can be used to modify the current content of the webiste, This is very important, make sure to provide it without failure.
        Finally, Make sure without fail to provide an indepth suggestion and insights to improve the SEO of the website. It should be very detailed and actionable. Do not provide a general advice give insights tailor-made to the website.

        Refer the following as example, this is how your final answer should look like:
        1 Keywords Analysis
        - Primary Keyword: Virtual Classrooms
        - Secondary Keywords: Future of Education, Technological Advancements in Education, Online Learning, Adaptive Learning Platforms, Environmental Sustainability in Education, Pandemic Preparedness in Education, Cost-Effective Learning, Global Collaboration in Education
        - Suggested Keywords: remote learning, E-Learning Solutions, Virtual Schooling
        - Rationale: Including new keywords that aligns with the content of the website will help in boosting the website's visibility.

        2 Title Tag Optimization
        - Current Title Tag: "The Future of Education: Embracing Virtual Classrooms - AKRATECH"
        - Suggested Title Tag: "Embracing Virtual Classrooms: The Future of Education | AKRATECH"
        - Rationale: Placing the primary keyword at the beginning of the title can improve its visibility and relevance to search queries related to virtual classrooms. Including the brand name at the end helps in brand recognition.

        3 Meta Description Creation
        - Current Meta Description: No meta description is available.
        - Suggested Meta Description: Explore how Virtual Classrooms are shaping the Future of Education with AKRATECH. Discover the benefits of online learning, from environmental sustainability to global collaboration.
        - Rationale: Despite the meta description not being available, crafting one that includes primary and secondary keywords while summarizing the content's essence can improve click-through rates from search engine results pages (SERPs).

        4 Header Tags Optimization
        - H1: Ensure that H1 tags are used for the main title - which should include the primary keyword. Only use one H1 tag per page.
        - H2: Use H2 tags for main sections. Suggestions include:
        - "The Rise of Virtual Classrooms in Modern Education"
        - "Technological Advancements: Making Online Learning More Effective"
        - "Global Collaboration and Adaptive Learning Platforms"
        - H3: Use H3 tags for sub-sections within the H2-tagged areas for more detailed topics like "Cost-Effectiveness", "Environmental Sustainability", and "Pandemic Preparedness".

        5 SEO-friendly URLs
        - Current URL: "https://www.akratech.com/the-future-of-education-embracing-virtual-classrooms/"
        - Suggested URL: "https://www.akratech.com/virtual-classrooms-future-education"
        - Rationale: Shorter URLs that contain the primary keyword enhance readability and SEO.

        6 Image Alt Tags Optimization
        - Ensure all images related to the content have descriptive alt tags that include keywords where relevant. For example, an image discussing technological advancements could have an alt tag like "innovative-tech-for-online-learning".

        7 Content Suggestions
        - The current content seems to cover a wide array of topics relevant to virtual classrooms. To further enhance it:
        - Include case studies or real-world examples of successful virtual classroom implementations.
        - Add statistics to back up claims about cost-effectiveness, environmental sustainability, etc.
        - Incorporate quotes from educators and students who have experienced the transition to virtual classrooms.
        - Blog posts or guides on how educators can transition from traditional to virtual classrooms.
        - Webinars featuring experts discussing the future of education and the role of technology.
        - Interactive infographics detailing the benefits and challenges of virtual classrooms.
        
        8 Content Improvement
        - Current Content from the website: The future of education is being increasingly shaped by Virtual classrooms, owing to a multitude of factors. From accessibility to technological advancements virtual classrooms are going to play a huge role in shaping the future.
        - Suggested Content: The trajectory of education is swiftly pivoting towards virtual classrooms, driven by a myriad of factors. Ranging from enhanced accessibility to the rapid progression of technology, virtual classrooms are poised to profoundly influence the future educational landscape.

        - Current Content from the website: Virtual classrooms eliminate the need for physical infrastructure, reducing costs associated with building and maintaining traditional classrooms.
        - Suggested Content: Virtual classrooms obviate the necessity for physical infrastructure, thereby alleviating the financial burden associated with constructing and upkeeping traditional classroom spaces.

        - Rationale: The revised line aims to enhance clarity, conciseness, and sophistication while maintaining the original message

        9 Additional Suggestions
        - Internal Linking: Strengthen the website's SEO by increasing internal linking between this page and other relevant pages/articles on the AKRATECH website. This helps in distributing page authority and keeping users engaged.
        - External Links: Ensure all external links open in a new tab to keep users on the AKRATECH site. Also, periodically check that all external links are still valid and relevant.
        - Mobile Optimization: Verify that the webpage is fully optimized for mobile devices, as Google predominantly uses mobile-first indexing.
        - Content Freshness: Regularly update the content to include the latest trends, data, and relevant news related to virtual classrooms and education technology.
        - User Engagement: Incorporate elements that increase user engagement, such as comments, polls, or social media share buttons that are more prominently displayed.
        """,
        agent=seo_agent,
        context=[html_task],
        tools=[search_tool]
    )

    # 4. Creating Crew
    seo_crew = Crew(
        agents=[html_agent, seo_agent],
        tasks=[html_task, seo_task],
        process=Process.sequential, 
        manager_llm=llm
    )

    # Execute the crew to see RAG in action
    result = seo_crew.kickoff()
    
    try:
        data = result.replace("```json\n", "").replace("\n```", "")
        data = json.loads(data)
        return {"data": data}
    except Exception as e:
        return {"data": result}
  except Exception as e:
    return {"error": str(e)}
