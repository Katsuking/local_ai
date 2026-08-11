import { McpServer } from "@modelcontextprotocol/server";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";
import { readdir, stat } from "node:fs/promises";
import { join } from "node:path";

/**
 * Initialize the MCP server.
 * Provide a name and version for the server which will be reported to the client.
 */
const server = new McpServer({
  name: "bun-mcp-server",
  version: "1.0.0",
});

/**
 * 1. Simple Tool: say-hello
 * Greets a user by their name.
 * Useful for verifying that the MCP client connection and communication are working correctly.
 */
server.registerTool(
  "say-hello",
  {
    description: "Greet a user by their name",
    inputSchema: z.object({
      name: z.string().describe("The name of the user to greet"),
    }),
  },
  async ({ name }) => {
    // Return a simple text greeting message
    return {
      content: [
        {
          type: "text",
          text: `Hello, ${name}! Welcome to the Bun MCP Server.`,
        },
      ],
    };
  }
);

/**
 * 2. Practical Tool: search-files
 * Searches for files with a specific extension in a local directory.
 * Returns metadata such as name, size, and last modified date for matched files.
 */
server.registerTool(
  "search-files",
  {
    description: "Search for files with a specific extension in a local directory and return metadata",
    inputSchema: z.object({
      directoryPath: z.string().describe("The absolute path of the directory to search"),
      extension: z.string().optional().describe("Filter files by extension (e.g., 'ts', 'md') without the dot"),
    }),
  },
  async ({ directoryPath, extension }) => {
    try {
      // Read all file/folder names in the specified directory
      const files = await readdir(directoryPath);
      const matchedFiles = [];

      for (const file of files) {
        // Apply extension filter if it is provided
        if (extension && !file.endsWith(`.${extension}`)) {
          continue;
        }

        const fullPath = join(directoryPath, file);
        const fileStat = await stat(fullPath);
        
        // Only return metadata if it's a file (skipping directories)
        if (fileStat.isFile()) {
          matchedFiles.push({
            name: file,
            size: `${(fileStat.size / 1024).toFixed(2)} KB`,
            modifiedAt: fileStat.mtime.toISOString(),
          });
        }
      }

      // Return the JSON list of matched files
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(matchedFiles, null, 2),
          },
        ],
      };
    } catch (error: any) {
      // Return an error message to the client in case of failure
      return {
        isError: true,
        content: [{ type: "text", text: `Failed to read directory: ${error.message}` }],
      };
    }
  }
);

/**
 * read-file
 * Reads the text content of a local file safely.
 * Includes a maxLength parameter to prevent sending huge files which can exceed token limits.
 */
server.registerTool(
  "read-file",
  {
    description: "Read the content of a local file with a size limit to avoid excessive token usage",
    inputSchema: z.object({
      filePath: z.string().describe("The absolute path of the file to read"),
      maxLength: z.number().optional().default(5000).describe("Maximum character length to read"),
    }),
  },
  async ({ filePath, maxLength }) => {
    try {
      // Check if the file exists using Bun's native file API
      const file = Bun.file(filePath);
      if (!(await file.exists())) {
        throw new Error(`File does not exist: ${filePath}`);
      }

      // Read file content as plain text
      let content = await file.text();
      
      // Limit file content length to prevent excessive token usage in LLMs
      if (content.length > maxLength) {
        content = content.substring(0, maxLength) + `\n\n... [Truncated! Only showing first ${maxLength} characters]`;
      }

      // Return the file content
      return {
        content: [
          {
            type: "text",
            text: content,
          },
        ],
      };
    } catch (error: any) {
      // Return an error message to the client in case of failure
      return {
        isError: true,
        content: [{ type: "text", text: `Failed to read file: ${error.message}` }],
      };
    }
  }
);

/**
 * Connect the server to the Stdio transport.
 * This establishes the stdin/stdout communication channel, typically used by MCP clients (like Claude Desktop).
 */
const transport = new StdioServerTransport();
await server.connect(transport);
