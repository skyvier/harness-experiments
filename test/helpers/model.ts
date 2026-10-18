import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { BindToolsInput } from "@langchain/core/language_models/chat_models";
import type { BaseMessage, AIMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";

/** Exercises the real LangChain graph without a provider or network calls. */
export class ScriptedModel extends BaseChatModel {
  readonly requests: BaseMessage[][] = [];
  boundTools: BindToolsInput[] = [];
  readonly #respond: (
    messages: BaseMessage[],
  ) => AIMessage | Promise<AIMessage>;

  constructor(
    respond: (messages: BaseMessage[]) => AIMessage | Promise<AIMessage>,
  ) {
    super({});
    this.#respond = respond;
  }

  _llmType(): string {
    return "scripted-test-model";
  }

  bindTools(tools: BindToolsInput[]): this {
    this.boundTools = tools;
    return this;
  }

  async _generate(messages: BaseMessage[]): Promise<ChatResult> {
    this.requests.push([...messages]);
    const message = await this.#respond(messages);
    return {
      generations: [
        {
          message,
          text: typeof message.content === "string" ? message.content : "",
        },
      ],
    };
  }
}
