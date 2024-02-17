import { NextRequest } from 'next/server';
import { Message as VercelChatMessage, StreamingTextResponse } from 'ai';
import { AIMessage, ChatMessage, HumanMessage } from "@langchain/core/messages";
import { pull } from "langchain/hub";

import { ChatOpenAI } from '@langchain/openai';
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";

import { BaseCallbackHandler } from "@langchain/core/callbacks/base";

import { kv } from '@vercel/kv'

import { auth } from '@/auth'
import { nanoid } from '@/lib/utils'
import { LLMResult } from '@langchain/core/outputs';

import { AgentExecutor, createStructuredChatAgent,createReactAgent } from "langchain/agents";
import { BingSerpAPI } from "@langchain/community/tools/bingserpapi";
import { Calculator } from "langchain/tools/calculator";
export const runtime = 'edge';

const convertVercelMessageToLangChainMessage = (message: VercelChatMessage) => {
  if (message.role === "user") {
    return new HumanMessage(message.content);
  } else if (message.role === "assistant") {
    return new AIMessage(message.content);
  } else {
    return new ChatMessage(message.content, message.role);
  }
};

class MyCallbackHandler extends BaseCallbackHandler {
  private body: any;
  private userId: string;
  constructor(requestBody: any, userId: string) {
    super();
    this.body = requestBody;
    this.userId = userId;
  }
  name = "MyCallbackHandler";
  async handleLLMEnd(output: LLMResult, runId: string, parentRunId?: string | undefined, tags?: string[] | undefined) {
    const title = this.body.messages[0].content.substring(0, 100)
    const id = this.body.id ?? nanoid()
    const createdAt = Date.now()
    const path = `/chat/${id}`
    const payload = {
      id,
      title,
      userId:this.userId,
      createdAt,
      path,
      messages: [
        ...this.body.messages,
        {
          content: output.generations[0][0].text,
          role: 'assistant'
        }
      ]
    }
    await kv.hmset(`chat:${id}`, payload)
    await kv.zadd(`user:chat:${this.userId}`, {
      score: createdAt,
      member: `chat:${id}`
    })
  }

  // async handleChainStart(chain: Serialized) {
  //   console.log(`Entering new ${chain.id} chain...`);
  // }

  // async handleChainEnd(_output: ChainValues) {
  //   console.log("Finished chain.");
  // }

  // async handleAgentAction(action: AgentAction) {
  //   console.log(action.log);
  // }

  // async handleToolEnd(output: string) {
  //   console.log(output);
  // }

  // async handleText(text: string) {
  //   console.log(text);
  // }

  // async handleAgentEnd(action: AgentFinish) {
  //   console.log(action);
  // }
}

export async function POST(req: NextRequest) {
  const userId = (await auth())?.user.id

  if (!userId) {
    return new Response('Unauthorized', {
      status: 401
    })
  }
  const body = await req.json();
  const messages = body.messages
  const previousMessages = messages
      .slice(0, -1)
      .map(convertVercelMessageToLangChainMessage);
  const currentMessageContent = messages[messages.length - 1].content;
  const tools = [
    new BingSerpAPI(process.env.BINGSERP_API_KEY)
  ];

  const SYSTEM_TEMPLATE = `Respond after accessing one of the tools {tools} with names "{tool_names}" if neccesary:
  {agent_scratchpad}
     \`\`\`
  {{
    "action": $TOOL_NAME,
    "action_input": $INPUT
  }}
  \`\`\`
  Observation: action result Then output final answer briefly in markdown format like  \`\`\`markdown\n {{final_answer}}  \`\`\`
  ended with a json blob\`\`\`
  {{
    "action":  "Final Answer",
    "action_input":  "DONE",
  }}
  \`\`\`
  The question is:\n`;
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", SYSTEM_TEMPLATE],
    ["human", "{input}"],
  ]);

  // const prompt = await pull<PromptTemplate>("hwchase17/react");
  
  const model = new ChatOpenAI({
    temperature: 0.1,
    modelName:process.env.LLM_MODEL,
    openAIApiKey:process.env.OPENAI_API_KEY,
    configuration:{
      baseURL:process.env.OPENAI_BASE_URL
    },
    streaming: true
  });
  
  const myCallback = new MyCallbackHandler(body, userId);
 
  // const outputParser = new BytesOutputParser();

  // const chain = prompt.pipe(model).pipe(outputParser);

 
  // const stream = await chain.stream({
  //   chat_history: formattedPreviousMessages.join('\n'),
  //   input: currentMessageContent,
  // },{
  //   callbacks: [myCallback]
  // });

  const agent = await createStructuredChatAgent({
    llm: model,
    tools,
    prompt,
  });

  const agentExecutor = new AgentExecutor({
    agent,
    tools,
    returnIntermediateSteps: false,
  });

  
  const logStream = await agentExecutor.streamLog({
    input: currentMessageContent,
    chat_history: previousMessages,
  },{
      callbacks: [myCallback]
  });

  const textEncoder = new TextEncoder();
  const transformStream = new ReadableStream({
    async start(controller) {
      for await (const chunk of logStream) {

        if (chunk.ops?.length > 0 && chunk.ops[0].op === "add") {
          const addOp = chunk.ops[0];
          if (
            addOp.path.startsWith("/logs/ChatOpenAI") &&
            typeof addOp.value === "string" &&
            addOp.value.length
          ) {
            controller.enqueue(textEncoder.encode(addOp.value));
          }
        }
      }
      controller.close();
    },
  });

  return new StreamingTextResponse(transformStream);
}