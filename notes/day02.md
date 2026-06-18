## Day 2 — The four primitives + the interaction loop
what is the difference between a tool and a prompt primitive in MCP??
Tools is list is tools provided by MCP servers to connect to external resources like API or Database or file system, for eg api request, connection/query to Database , access url for file.Prompts are set of resuable Templates available on MCP server which used by LLMs for a domain for a tasks for eg travel plan templates.
Tools are invoked by the agent (the LLM decides when to call them), while prompts are selected by the user/host and injected into context before the LLM even starts reasoning. That's a different trigger point, which means a different test target.

