from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo


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


multimodel_agent = Agent(
    model = Groq(id="llama-3.3-70b-versatile"),
    team = [web_search_agent, financial_agent],
    instructions=["always include sources in your response", "use tables to display stock data"],
    show_tools_calls=True,
    markdown=True,
)


multimodel_agent.print_response("summarize analyst recommndations and share the latest news for NVDA " , stream = True)