"""Generador de analítica visual y reportes ejecutivos para eval-harness.

Produce:
1. Serie de tiempo/entropía (N vs. Defectos/Colisión).
2. Scatterplot de Contención (PGDR vs. System Breach Rate).
3. Arquetipos conductuales de la tríada (Atajos vs. Colisión).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Operación headless sin display


def load_data(json_path: str) -> dict[str, Any]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def plot_entropy_series(slices: list[dict[str, Any]], out_dir: Path) -> None:
    """Serie de entropía: Degradación del PGDR a medida que aumenta N."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    models = sorted(list(set(s["model"] for s in slices)))
    colors = {
        "gpt-5.6-luna": "#E63946",
        "gpt-5.6-sol": "#2A9D8F",
        "gpt-5.6-terra": "#E76F51",
        "gpt-6-astra": "#457B9D",
    }
    markers = {"baseline_autorregresivo": "--o", "neurosymbolic_handoff": "-s"}

    for model in models:
        for cond in ["baseline_autorregresivo", "neurosymbolic_handoff"]:
            subset = [s for s in slices if s["model"] == model and s["condition"] == cond]
            subset.sort(key=lambda x: x["entropy"])
            x = [s["entropy"] for s in subset]
            y = [s["pgdr_pct"] for s in subset]
            label = f"{model} ({'Base' if 'base' in cond else 'NeuroSym'})"
            ax.plot(
                x,
                y,
                markers[cond],
                color=colors.get(model, "#333333"),
                label=label,
                linewidth=1.8,
                markersize=6,
            )

    ax.set_title("Curva de Degradación Atencional por Densidad de Herramientas ($N$)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Número de Herramientas en Contexto ($N$)", fontsize=10)
    ax.set_ylabel("Pre-Gate Defect Rate (%)", fontsize=10)
    ax.set_xticks([10, 50, 128])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "entropy_degradation_series.png")
    plt.close(fig)


def plot_containment_scatter(slices: list[dict[str, Any]], out_dir: Path) -> None:
    """Scatterplot: Demostración de la invariante Post-Gate Breach = 0.0%."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    for s in slices:
        is_neuro = s["condition"] == "neurosymbolic_handoff"
        color = "#2A9D8F" if is_neuro else "#E63946"
        marker = "s" if is_neuro else "o"
        ax.scatter(
            s["pgdr_pct"],
            s["system_breach_pct"],
            color=color,
            marker=marker,
            s=s["entropy"] * 1.5,
            alpha=0.75,
            edgecolors="black",
            linewidth=0.5,
        )

    # Línea de equivalencia crítica (Sin protección)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Línea de falla total ($Breach = Defect$)")
    ax.axhline(0, color="#2A9D8F", linestyle="-", linewidth=2, label="Invariante Neuro-Simbólica ($Breach = 0.0\\%$)")

    ax.set_title("Contención Determinista: Defectos Intrínsecos vs. Brechas de Sistema", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Pre-Gate Defect Rate del Modelo (%)", fontsize=10)
    ax.set_ylabel("Post-Gate System Breach Rate (%)", fontsize=10)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 65)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "containment_scatter.png")
    plt.close(fig)


def plot_behavioral_archetypes(slices: list[dict[str, Any]], out_dir: Path) -> None:
    """Comparativa de perfiles: Atajo (Short-Circuit) vs. Colisión Léxica (SCR)."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    models = sorted(list(set(s["model"] for s in slices)))
    x = range(len(models))
    width = 0.35

    # Tomamos el caso N=10 para atajos y N=128 para colisiones
    short_circuits = []
    collisions = []

    for m in models:
        s_n10 = next(s for s in slices if s["model"] == m and s["condition"] == "baseline_autorregresivo" and s["entropy"] == 10)
        s_n128 = next(s for s in slices if s["model"] == m and s["condition"] == "baseline_autorregresivo" and s["entropy"] == 128)
        short_circuits.append(s_n10["short_circuit_attempt_rate"] * 100)
        collisions.append(s_n128["syntax_collision_rate"] * 100)

    ax.bar([i - width / 2 for i in x], short_circuits, width, label="Propensión a Atajo (SCAR @ N=10)", color="#E76F51")
    ax.bar([i + width / 2 for i in x], collisions, width, label="Vulnerabilidad a Señuelos (SCR @ N=128)", color="#457B9D")

    title = "Arquetipos Conductuales de la Tríada GPT-5.6" if len(models) == 3 else "Arquetipos Conductuales por Modelo"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Tasa de Ocurrencia (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "behavioral_archetypes.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generador visual de reportes analíticos.")
    parser.add_argument("--input", default="results/benchmark_summary_real.json", help="Ruta al JSON de métricas.")
    parser.add_argument("--out-dir", default="results/figures", help="Directorio destino para las figuras.")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = load_data(args.input)
    slices = data.get("slices", [])

    print(f"-> Procesando {len(slices)} cortes experimentales desde {args.input}...")
    plot_entropy_series(slices, out_path)
    plot_containment_scatter(slices, out_path)
    plot_behavioral_archetypes(slices, out_path)

    print(f"✅ Figuras generadas con éxito en: {out_path.resolve()}/")
    print("  • entropy_degradation_series.png")
    print("  • containment_scatter.png")
    print("  • behavioral_archetypes.png")


if __name__ == "__main__":
    main()
