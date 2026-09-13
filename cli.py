"""Interactive terminal chat over your local documents, via Lemonade Server."""
from rich.console import Console
from rich.markdown import Markdown

from rag import LemonRAG
import config

console = Console()


def main():
    console.print("[bold yellow]🍋 LemonRAG[/bold yellow] — local doc chat, powered by Lemonade")
    console.print(f"[dim]model: {config.LEMONADE_MODEL}  |  endpoint: {config.LEMONADE_BASE_URL}[/dim]")

    try:
        rag = LemonRAG()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    console.print("[dim]Type a question, or 'quit' to exit.[/dim]\n")

    while True:
        try:
            query = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        try:
            result = rag.answer(query)
        except Exception as e:
            console.print(f"[red]Error talking to Lemonade Server: {e}[/red]")
            console.print(
                "[dim]Is Lemonade Server running? Try: lemonade run "
                f"{config.LEMONADE_MODEL}[/dim]"
            )
            continue

        console.print()
        console.print(Markdown(result["answer"]))
        if result["sources"]:
            console.print(f"\n[dim]Sources: {', '.join(result['sources'])}[/dim]")
        console.print()


if __name__ == "__main__":
    main()
