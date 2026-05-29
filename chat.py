"""
Terminal-based chat interface for the Browser Agent.
Use this if you prefer CLI over the Gradio web UI.
"""

import asyncio
import sys
from agent import BrowserAgentSession, get_llm


BANNER = """
╔══════════════════════════════════════════════════════╗
║          🌐  Browser Agent — LM Studio  🌐          ║
║                                                      ║
║  Type a task and the agent will control Chrome.      ║
║  Type 'close' to close the browser.                 ║
║  Type 'quit' or 'exit' to stop.                     ║
╚══════════════════════════════════════════════════════╝
"""


async def main():
    print(BANNER)

    llm = get_llm()
    print(f"  Model:  {llm.model}")
    print(f"  Server: {llm.base_url}")
    print()

    session = BrowserAgentSession(llm=llm)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nClosing browser and exiting...")
            await session.close_browser()
            print("Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            await session.close_browser()
            print("Goodbye!")
            break

        if user_input.lower() == "close":
            if session.is_browser_open:
                await session.close_browser()
                print("✅ Browser closed.\n")
            else:
                print("ℹ️  No browser is currently open.\n")
            continue

        print("\n⏳ Working on it... Watch the browser window.\n")

        try:
            result = await session.run_task(task=user_input)
            print(f"\n✅ Result:\n{result}")
            print(
                "\n🔍 Browser is still open — verify the result."
                "\n   Type 'close' to close it, or send another task.\n"
            )
        except KeyboardInterrupt:
            print("\n⚠️  Task cancelled.\n")
            await session.close_browser()
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg:
                print(
                    "\n❌ Cannot connect to LM Studio.\n"
                    "   Make sure LM Studio is running with a model loaded.\n"
                )
            else:
                print(f"\n❌ Error: {error_msg}\n")


if __name__ == "__main__":
    asyncio.run(main())
