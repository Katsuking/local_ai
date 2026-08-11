import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import ollama from "ollama";
import { join } from "node:path";

// 1. Initialize the MCP Client.
// Define name and version which will be shared with the MCP server.
const mcpClient = new Client(
  {
    name: "ollama-mcp-client",
    version: "1.0.0",
  },
  {
    capabilities: {},
  }
);

// 2. Configure transport to launch the sibling MCP server.
// Using 'bun run' to execute server/index.ts directly.
const transport = new StdioClientTransport({
  command: "bun",
  args: ["run", join(import.meta.dir, "../server/index.ts")],
});

console.log("Connecting to MCP server...");
await mcpClient.connect(transport);
console.log("Connected to MCP server successfully!");

try {
  // 3. Fetch list of tools supported by the MCP server.
  const toolsResponse = await mcpClient.listTools();
  
  // 4. Map the MCP tools to Ollama's tool definition schema.
  // Cast as any[] to satisfy Ollama's strict parameters type check.
  const ollamaTools = toolsResponse.tools.map((tool) => ({
    type: "function" as const,
    function: {
      name: tool.name,
      description: tool.description || "",
      parameters: tool.inputSchema,
    },
  })) as any[];

  // Define a prompt that requires using local file tools (searching and reading files).
  const userMessage = "'/home/mojo/Documents/dev/local_ai/mcp_bun/server' directory ディレクトリにどんなファイルがあるか教えて、またどんなツールが登録されているか教えて";
  
  const messages: any[] = [
    {
      role: "user",
      content: userMessage,
    },
  ];

  console.log(`\nUser prompt: "${userMessage}"`);
  console.log("Sending prompt to Ollama (qwen3:8b) with tools capability...");

  // 5. Query Ollama model with tools configuration.
  // Explicitly set stream: false to keep return type consistent.
  let response = await ollama.chat({
    model: "qwen3:8b",
    messages: messages,
    tools: ollamaTools,
    stream: false,
  });

  // 6. Check if the LLM decided to call any tools.
  if (response.message.tool_calls && response.message.tool_calls.length > 0) {
    // Append the assistant message (containing tool calls) to history.
    console.log(response.message + "<- message")
    messages.push(response.message);

    for (const toolCall of response.message.tool_calls) {
      const { name, arguments: args } = toolCall.function;
      console.log(`\n[Agent Action] LLM decided to call tool "${name}" with args:`, args);

      // Execute the tool on our MCP server.
      const toolResult = await mcpClient.callTool({
        name,
        arguments: args,
      });
      console.log(`[Agent Response] Tool "${name}" executed successfully.`);

      // Convert tool results (usually text content blocks) into a single string.
      const textResult = toolResult.content
        .filter((c) => c.type === "text")
        .map((c: any) => c.text)
        .join("\n");

      // Push the tool result back into the chat history as a 'tool' role.
      messages.push({
        role: "tool",
        content: textResult,
        name: name,
      });
    }

    // 7. Request final answer from Ollama using the gathered tool results.
    console.log("\nGenerating final response from Ollama...");
    response = await ollama.chat({
      model: "qwen3:8b",
      messages: messages,
      stream: false,
    });
  }

  // 8. Output the final consolidated response.
  console.log("\n--- Final Response from LLM ---");
  console.log(response.message.content);

} catch (error) {
  console.error("An error occurred during agent execution:", error);
} finally {
  // Always close connection to terminate the server subprocess cleanly.
  await transport.close();
}