import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { McpServer, fromJsonSchema, WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/server";
import { join } from "node:path";

// 1. Initialize the Stdio Client to connect to the sibling MCP server (server/index.ts).
// This client will act as a bridge, forwarding tool execution requests to the stdio server.
const mcpClient = new Client(
  {
    name: "ollama-mcp-client",
    version: "1.0.0",
  },
  {
    capabilities: {},
  }
);

// Configure the stdio transport to launch server/index.ts using Bun.
const transport = new StdioClientTransport({
  command: "bun",
  args: ["run", join(import.meta.dir, "../server/index.ts")],
});

console.log("Connecting to sibling MCP server...");
await mcpClient.connect(transport);
console.log("Connected to sibling MCP server successfully!");

// 2. Fetch list of tools supported by the sibling MCP server.
const toolsResponse = await mcpClient.listTools();

// 3. Initialize the SSE MCP Server that will be exposed to Open WebUI.
const sseMcpServer = new McpServer({
  name: "sse-mcp-server",
  version: "1.0.0",
});

// 4. Register all tools from the sibling MCP server to the SSE MCP server.
// We dynamically map the stdio server's tools so Open WebUI can discover them.
for (const tool of toolsResponse.tools) {
  console.log(`Registering tool bridge for: ${tool.name}`);
  sseMcpServer.registerTool(
    tool.name,
    {
      description: tool.description || "",
      // Convert the plain JSON schema returned by listTools to the type required by McpServer.
      inputSchema: fromJsonSchema(tool.inputSchema as any),
    },
    async (args) => {
      console.log(`[Bridge Action] Forwarding call to tool "${tool.name}" with args:`, args);
      // Forward the execution request to the sibling stdio server.
      // Cast args as Record<string, any> to satisfy TypeScript's type check for callTool arguments.
      const result = await mcpClient.callTool({
        name: tool.name,
        arguments: args as Record<string, any>,
      });
      return result;
    }
  );
}

// 5. Initialize the Streamable HTTP (SSE) Server Transport.
// Open WebUI communicates using this standard transport.
const sseTransport = new WebStandardStreamableHTTPServerTransport({
  // Generate a random UUID to manage client sessions statelessly/statefully.
  sessionIdGenerator: () => crypto.randomUUID(),
});

// Connect the SSE server to the Streamable HTTP transport.
await sseMcpServer.connect(sseTransport);

// 6. Start the HTTP server using Bun's native server API.
const port = 3001;
Bun.serve({
  port,
  async fetch(req) {
    // Handle CORS preflight requests (required for browser-based clients like Open WebUI).
    if (req.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS, DELETE",
          "Access-Control-Allow-Headers": "Content-Type, Authorization, mcp-protocol-version",
        },
      });
    }

    // Process the request using the Streamable HTTP transport handler.
    const response = await sseTransport.handleRequest(req);
    
    // Clone and append necessary CORS headers to the response before returning.
    const newHeaders = new Headers(response.headers);
    newHeaders.set("Access-Control-Allow-Origin", "*");
    newHeaders.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE");
    newHeaders.set("Access-Control-Allow-Headers", "Content-Type, Authorization, mcp-protocol-version");
    
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: newHeaders,
    });
  },
});

console.log(`\n[Success] SSE MCP Bridge Server running at http://localhost:${port}/mcp`);
console.log(`To connect Open WebUI, configure an MCP server of type "Streamable HTTP" pointing to:`);
console.log(`http://localhost:${port}/mcp`);