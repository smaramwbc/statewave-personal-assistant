"""Seed script — populates Statewave with demo user profiles.

Usage:
    python -m scripts.seed
    # or, after pip install -e .[dev]:
    pa-seed

Each profile's episodes are ingested first, then memories are compiled.
Statewave extracts structured memory facts from the episode payloads automatically.
"""

import asyncio
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from app.core.config import settings
from app.data.profiles import PROFILES
from app.services.statewave import StatewaveClient, StatewaveError

console = Console()
logger = logging.getLogger(__name__)


async def seed_user(sw: StatewaveClient, profile: dict) -> dict:
    user_id = profile["user_id"]
    results = {"user_id": user_id, "episodes": 0, "memories_compiled": 0, "errors": 0}

    # Ingest episodes (Statewave compiles memories from these)
    for ep in profile.get("episodes", []):
        try:
            await sw.record_episode(
                subject_id=user_id,
                user_message=ep["user_message"],
                assistant_response=ep["assistant_response"],
            )
            results["episodes"] += 1
        except StatewaveError as exc:
            logger.warning("Episode record failed for %s: %s", user_id, exc)
            results["errors"] += 1

    # Compile memories from the ingested episodes
    if results["episodes"] > 0:
        try:
            compiled = await sw.compile_memories(user_id)
            results["memories_compiled"] = compiled.memories_created
        except StatewaveError as exc:
            logger.warning("Memory compilation failed for %s: %s", user_id, exc)
            results["errors"] += 1

    return results


async def _run() -> None:
    console.rule("[bold blue]Statewave Memory Quickstart — Seed Script[/]")
    console.print(f"Target: [cyan]{settings.statewave_base_url}[/]")
    console.print(f"Seeding [bold]{len(PROFILES)}[/] demo users...\n")

    async with StatewaveClient() as sw:
        all_results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for profile in PROFILES:
                task = progress.add_task(
                    f"Seeding {profile['user_id']} — {profile['display_name']}...",
                    total=None,
                )
                result = await seed_user(sw, profile)
                all_results.append((profile, result))
                progress.update(task, completed=True)

    table = Table(title="Seed Results", show_lines=True)
    table.add_column("User ID", style="cyan")
    table.add_column("Name")
    table.add_column("Episodes", justify="right")
    table.add_column("Memories Compiled", justify="right")
    table.add_column("Errors", justify="right", style="red")

    for profile, res in all_results:
        table.add_row(
            res["user_id"],
            profile["display_name"],
            str(res["episodes"]),
            str(res["memories_compiled"]),
            str(res["errors"]) if res["errors"] else "[green]0[/]",
        )

    console.print(table)
    console.print("\n[bold green]Done.[/] Open [cyan]http://localhost:8000[/] after starting the server.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
