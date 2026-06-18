Review the code, answer these five questions. No looking at the file:

What does mask_error_details=True do and why does it matter for a QA engineer testing this server?
Ans : this helps to handle unexpected errors that might occur like db down or keyerror , if such issues comes python will push these errors in stack trace which might contain sensitive information like db credentials or server paths which will be available to LLM client and hence ai might read the same and pass it to user.by masking fastmcp will intercepts such unhandled erros and replace it with common error message. It is needed by QA engineers to verify that sensitive information should not be passed to ai and in turn user. 

Why does every print() statement have file=sys.stderr? What breaks if you remove it?
Ans : Writing to stdout will corrupt the JSON-RPC messages and break the server.while the print function writes to stdout by default,  uses file=sys.stderr safely.

ToolError is raised in multiple places — what's the difference between raising ToolError vs letting a Python exception propagate naturally?
Ans: oolError transforms/present errors to AI in a readable format so that AI can understand and take action on it while simple python errors will confuse AI and it can hallucinate an apology or give up. With ToolError we can also handle runtime issues wherein response is formatted as a compliant Json package without breaking the host system.and the JSON-compliant response point is the key one for testing. You'll assert on this in Day 8.

In get_weather_forecast, days has a default of 3 and a validation check 1 <= days <= 3. What failure mode does this catch that the schema alone wouldn't?
Ans: min max value validation.The inputSchema defines the type as int but doesn't enforce range constraints. So a client sending days=10 passes schema validation but breaks the business logic. That gap between schema-valid and business-valid is a whole category of test cases you'll write in Phase 2.

Why is mcp.run() inside if __name__ == "__main__": and not just called at the top level?
Ans: python rule where it assigns special string main to name variable if conditon evaluates to true mcp.run boots up the server