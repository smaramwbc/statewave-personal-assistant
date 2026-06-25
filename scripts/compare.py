"""Side-by-side CLI comparison: stateless assistant vs Statewave-backed assistant.

This is the primary demo moment. Run it and the terminal output diff is
immediately obvious.

Usage:
    python -m scripts.compare compare --user dev_alice --message "What was the issue we were debugging last week?"
    python -m scripts.compare budget --user dev_bob
    python -m scripts.compare chat --user dev_alice
"""

import asyncio
import textwrap
from typing import Annotated

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.core.config import settings
from app.services.llm import LLMService
from app.services.statewave import StatewaveClient, StatewaveError

app = typer.Typer(help="Compare stateless vs memory-backed AI assistant responses.")
console = Console()

_WRAP = 60  # chars per panel column


def _wrap(text: str) -> str:
    return "\n".join(textwrap.fill(line, _WRAP) if line else "" for line in text.splitlines())


def _check_config() -> None:
    if not settings.llm_configured:
        console.print(
            "[bold red]Missing env var:[/] LLM_API_KEY\n"
            "Copy [cyan].env.example[/] → [cyan].env[/] and fill it in."
        )
        raise typer.Exit(1)


async def _stateless_response(message: str, llm: LLMService) -> str:
    """Call the LLM with no memory context at all."""
    return await llm.chat(message, assembled_context="")


async def _memory_response(
    user_id: str, message: str, llm: LLMService, sw: StatewaveClient
) -> tuple[str, int, int]:
    """Call the LLM with Statewave context injected. Returns (reply, token_estimate, n_memories)."""
    try:
        bundle = await sw.get_context(user_id, task=message)
    except StatewaveError as exc:
        console.print(f"[red]Statewave error:[/] {exc}")
        raise typer.Exit(1) from exc

    reply = await llm.chat(message, bundle.assembled_context)
    return reply, bundle.token_estimate, len(bundle.memories)


def _side_by_side(
    message: str,
    stateless: str,
    with_memory: str,
    user_id: str,
    token_estimate: int,
    n_memories: int,
) -> None:
    console.print()
    console.rule(f"[bold]User message[/] — [cyan]{user_id}[/]")
    console.print(Panel(message, style="dim"))
    console.print()

    left = Panel(
        _wrap(stateless),
        title="[bold red]✗ Stateless assistant[/]",
        subtitle="No memory — blank slate every session",
        border_style="red",
        width=_WRAP + 6,
    )
    right = Panel(
        _wrap(with_memory),
        title="[bold green]✓ Statewave-backed assistant[/]",
        subtitle=f"{n_memories} memories · ~{token_estimate} tokens",
        border_style="green",
        width=_WRAP + 6,
    )
    console.print(Columns([left, right], equal=True))
    console.print()


async def _run_compare(user_id: str, message: str) -> None:
    _check_config()
    llm = LLMService()

    console.print(f"\n[dim]Querying both assistants for [cyan]{user_id}[/]...[/]")

    async with StatewaveClient() as sw:
        stateless, (with_memory, token_estimate, n_memories) = await asyncio.gather(
            _stateless_response(message, llm),
            _memory_response(user_id, message, llm, sw),
        )

    _side_by_side(message, stateless, with_memory, user_id, token_estimate, n_memories)


async def _run_budget_demo(user_id: str) -> None:
    """Demo 3: token budget enforcement."""
    _check_config()
    console.print()
    console.rule("[bold blue]Token Budget Enforcement Demo[/]")
    console.print(
        f"Fetching context for [cyan]{user_id}[/] at three different token budgets...\n"
    )

    table = Table(show_lines=True)
    table.add_column("max_tokens", justify="right", style="cyan")
    table.add_column("token_estimate", justify="right")
    table.add_column("memories_returned", justify="right")
    table.add_column("assembled_context (preview)", max_width=60)

    async with StatewaveClient() as sw:
        for budget in [200, 500, settings.statewave_max_tokens]:
            try:
                bundle = await sw.get_context(user_id, max_tokens=budget)
            except StatewaveError as exc:
                console.print(f"[red]{exc}[/]")
                continue

            preview = (bundle.assembled_context[:120] + "…") if bundle.assembled_context else "—"
            table.add_row(
                str(budget),
                str(bundle.token_estimate),
                str(len(bundle.memories)),
                preview,
            )

    console.print(table)
    console.print(
        "\n[dim]Compare with naive history injection:[/] dev_bob has 6 sessions → "
        "[bold red]~2,800 tokens of unranked noise[/] vs "
        f"[bold green]~{settings.statewave_max_tokens} tokens of ranked signal[/].\n"
    )


async def _run_memory_inspect(user_id: str) -> None:
    """Demo 2: memory state inspection."""
    _check_config()

    async with StatewaveClient() as sw:
        try:
            state = await sw.list_memories(user_id)
        except StatewaveError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc

    console.print()
    console.rule(f"[bold blue]Memory State — {user_id}[/]")
    console.print(
        f"[bold]{state.total_memories}[/] memories compiled. "
        + "  ".join(f"[cyan]{k}[/]: {v}" for k, v in state.memories_by_type.items())
    )
    console.print()

    for entry in state.entries:
        confidence_color = "green" if entry.confidence >= 0.95 else "yellow"
        header = Text()
        header.append(f"[{entry.kind}]  ", style="bold magenta")
        header.append(f"confidence={entry.confidence:.2f}", style=confidence_color)
        if entry.source_episode_ids:
            header.append(f"  source={entry.source_episode_ids[0]}", style="dim")
        if entry.tags:
            header.append(f"  tags={entry.tags}", style="dim cyan")

        console.print(Panel(f"{header}\n\n{entry.content}", border_style="dim"))


async def _run_interactive(user_id: str) -> None:
    """Interactive chat session against the Statewave-backed assistant."""
    _check_config()
    llm = LLMService()

    console.print()
    console.rule(f"[bold green]Interactive Chat — {user_id}[/]")
    console.print("[dim]Type [bold]quit[/] to exit. Memory context is loaded from Statewave.[/]\n")

    async with StatewaveClient() as sw:
        while True:
            try:
                message = console.input("[bold cyan]You:[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not message or message.lower() in {"quit", "exit", "q"}:
                break

            bundle = await sw.get_context(user_id, task=message)
            reply = await llm.chat(message, bundle.assembled_context)

            # Record the turn
            await sw.record_episode(
                subject_id=user_id,
                user_message=message,
                assistant_response=reply,
            )

            console.print(
                Panel(
                    reply,
                    title=f"[green]Assistant[/] · {len(bundle.memories)} memories · "
                    f"~{bundle.token_estimate} tokens",
                )
            )
            console.print()


# ── CLI commands ──────────────────────────────────────────────────────────────


@app.command("compare")
def cmd_compare(
    user: Annotated[str, typer.Option("--user", "-u", help="User ID to compare")] = "dev_alice",
    message: Annotated[
        str,
        typer.Option("--message", "-m", help="Message to send to both assistants"),
    ] = "What was the issue we were debugging last week?",
) -> None:
    """Side-by-side: stateless vs Statewave-backed assistant (Demo 1)."""
    asyncio.run(_run_compare(user, message))


@app.command("inspect")
def cmd_inspect(
    user: Annotated[str, typer.Option("--user", "-u")] = "dev_alice",
) -> None:
    """Inspect compiled memory state for a user (Demo 2)."""
    asyncio.run(_run_memory_inspect(user))


@app.command("budget")
def cmd_budget(
    user: Annotated[str, typer.Option("--user", "-u")] = "dev_bob",
) -> None:
    """Token budget enforcement demo — show how Statewave fits memory into limits (Demo 3)."""
    asyncio.run(_run_budget_demo(user))


@app.command("chat")
def cmd_chat(
    user: Annotated[str, typer.Option("--user", "-u")] = "dev_alice",
) -> None:
    """Interactive Statewave-backed chat session."""
    asyncio.run(_run_interactive(user))


if __name__ == "__main__":
    app()
