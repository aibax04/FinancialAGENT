from phi.agent import Agent 
import phi.api
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
import os
import phi  
from phi.playground import Playground , serve_playground_app


phi.api = os.getenv("PHI_API_KEY")

web_search_agent = Agent(
    name = "web search agent",
    role = "search the web for the information" ,
    model = Groq(id="llama-3.3-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions=["always include sources in your response"],
    show_tools_calls=True,
    markdown=True,
)


financial_agent = Agent(
    name = "financial agent",
    role = "search the web for financial information and answer questions about stocks, companies, and financial markets",
    model = Groq(id="llama-3.3-70b-versatile"),
    tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True , company_news=True), DuckDuckGo()],
    instructions=["use tables to display stock data, always include sources in your response"],
    show_tools_calls=True,
    markdown=True,
) 

app = Playground(agents= [financial_agent,web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True , port=7777)

