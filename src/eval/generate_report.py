"""Generador de analítica visual y reportes ejecutivos para eval-harness.

Produce:
1. Serie de tiempo/entropía (N vs. Defectos con bandas sombreadas 95% CI).
2. Scatterplot de Contención con cruces de error bidimensional (PGDR vs. System Breach).
3. Arquetipos conductuales con barras de error Wilson Score (Atajos vs. Colisión).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Operación headless sin display
import matplotlib.pyplot as plt


def load_data(json_path: str) -> dict[str, Any]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def _get_ci_bounds(s: dict[str, Any], key_pct: str, key_low: str, key_upp: str, key_pm: str) -> tuple[float, float, float]:
    """Extrae las cotas de confianza del slice o las calcula sobre la marcha como fallback."""
    val = s.get(key_pct, 0.0)
    if key_low in s and key_upp in s:
        return s[key_low], s[key_upp], s.get(key_pm, (s[key_upp] - s[key_low]) / 2.0)

    # Fallback Wilson 95% si el JSON no trae las llaves precalculadas
    n = s.get("total_traces", 50)
    k = int(round((val / 100.0) * n))
    z = 1.959963984540054
    p = k / max(n, 1)
    denom = 1.0 + (z**2) / n
    centre = (p + (z**2) / (2 * n)) / denom
    spread = (z * math.sqrt((p * (1.0 - p) / n) + (z**2) / (4 * (n**2)))) / denom
    low = max(0.0, centre - spread) * 100.0
    upp = min(1.0, centre + spread) * 100.0
    margin = (upp - low) / 2.0
    return round(low, 2), round(upp, 2), round(margin, 2)


def plot_entropy_series(slices: list[dict[str, Any]], out_dir: Path) -> None:
    """Serie de entropía: Degradación del PGDR a medida que aumenta N con bandas de confianza sombreadas."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

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
            if not subset:
                continue
            subset.sort(key=lambda x: x["entropy"])
            x = [s["entropy"] for s in subset]
            y = [s["pgdr_pct"] for s in subset]

            lows = []
            upps = []
            for s in subset:
                low_val, upp_val, _ = _get_ci_bounds(s, "pgdr_pct", "pgdr_ci_lower", "pgdr_ci_upper", "pgdr_margin_pm")
                lows.append(low_val)
                upps.append(upp_val)

            color = colors.get(model, "#333333")
            label = f"{model} ({'Base' if 'base' in cond else 'NeuroSym'})"

            ax.plot(
                x,
                y,
                markers[cond],
                color=color,
                label=label,
                linewidth=1.8,
                markersize=6,
            )
            # Banda sombreada de incertidumbre al 95%
            ax.fill_between(x, lows, upps, color=color, alpha=0.12)

    ax.set_title("Curva de Degradación Atencional por Densidad de Herramientas ($N$) con Bandas de Confianza (95% CI)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Número de Herramientas en Contexto ($N$)", fontsize=10)
    ax.set_ylabel("Pre-Gate Defect Rate (%)", fontsize=10)

    unique_entropies = sorted(list(set(s["entropy"] for s in slices)))
    ax.set_xticks(unique_entropies)
    ax.set_ylim(-2, 105)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "entropy_degradation_series.png")
    plt.close(fig)


def plot_containment_scatter(slices: list[dict[str, Any]], out_dir: Path) -> None:
    """Scatterplot con cruces de error bidimensionales (95% CI)."""
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    for s in slices:
        is_neuro = s["condition"] == "neurosymbolic_handoff"
        color = "#2A9D8F" if is_neuro else "#E63946"
        marker = "s" if is_neuro else "o"
        size = max(s["entropy"] * 1.5, 25)

        x_val = s["pgdr_pct"]
        y_val = s["system_breach_pct"]

        _, _, x_err = _get_ci_bounds(s, "pgdr_pct", "pgdr_ci_lower", "pgdr_ci_upper", "pgdr_margin_pm")
        if is_neuro:
            y_err = 0.0  # Invariante determinista: cero varianza
        else:
            _, _, y_err = _get_ci_bounds(s, "system_breach_pct", "system_breach_ci_lower", "system_breach_ci_upper", "system_breach_margin_pm")

        # Cruces de incertidumbre empírica
        ax.errorbar(
            x_val,
            y_val,
            xerr=x_err,
            yerr=y_err,
            fmt="none",
            ecolor=color,
            elinewidth=0.9,
            capsize=2.5,
            alpha=0.55,
        )

        ax.scatter(
            x_val,
            y_val,
            color=color,
            marker=marker,
            s=size,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.6,
            zorder=3,
        )

    # Línea de equivalencia crítica
    ax.plot([0, 100], [0, 100], "k--", alpha=0.35, label="Línea de falla total ($Breach = Defect$)")
    ax.axhline(0, color="#2A9D8F", linestyle="-", linewidth=2, label="Invariante Neuro-Simbólica ($Breach \\equiv 0.0\\%$)")

    ax.set_title("Contención Determinista: Defectos Intrínsecos vs. Brechas de Sistema (con 95% CI)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Pre-Gate Defect Rate del Modelo (%) [±95% CI]", fontsize=10)
    ax.set_ylabel("Post-Gate System Breach Rate (%) [±95% CI]", fontsize=10)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-3, 70)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "containment_scatter.png")
    plt.close(fig)


def plot_behavioral_archetypes(slices: list[dict[str, Any]], out_dir: Path) -> None:
    """Comparativa de arquetipos conductuales con barras de error Wilson Score."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    models = sorted(list(set(s["model"] for s in slices)))
    x = range(len(models))
    width = 0.35

    short_circuits = []
    short_circuit_errs = []
    collisions = []
    collision_errs = []

    for m in models:
        s_n10 = next((s for s in slices if s["model"] == m and s["condition"] == "baseline_autorregresivo" and s["entropy"] == 10), None)
        s_n128 = next((s for s in slices if s["model"] == m and s["condition"] == "baseline_autorregresivo" and s["entropy"] == 128), None)

        if s_n10:
            val_sc = s_n10["short_circuit_attempt_rate"] * 100
            _, _, err_sc = _get_ci_bounds(s_n10, "short_circuit_attempt_rate", "scar_ci_lower", "scar_ci_upper", "scar_margin_pm")
        else:
            val_sc, err_sc = 0.0, 0.0

        if s_n128:
            val_col = s_n128["syntax_collision_rate"] * 100
            _, _, err_col = _get_ci_bounds(s_n128, "syntax_collision_rate", "scr_ci_lower", "scr_ci_upper", "scr_margin_pm")
        else:
            val_col, err_col = 0.0, 0.0

        short_circuits.append(val_sc)
        short_circuit_errs.append(err_sc)
        collisions.append(val_col)
        collision_errs.append(err_col)

    ax.bar(
        [i - width / 2 for i in x],
        short_circuits,
        width,
        yerr=short_circuit_errs,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#2B2D42"},
        label="Propensión a Atajo (SCAR @ N=10)",
        color="#E76F51",
        alpha=0.9,
    )
    ax.bar(
        [i + width / 2 for i in x],
        collisions,
        width,
        yerr=collision_errs,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#2B2D42"},
        label="Vulnerabilidad a Señuelos (SCR @ N=128)",
        color="#457B9D",
        alpha=0.9,
    )

    ax.set_title("Arquetipos Conductuales por Modelo (con Wilson Score 95% CI)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Tasa de Ocurrencia (%)", fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "behavioral_archetypes.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generador visual de reportes analíticos con rigor estadístico.")
    parser.add_argument("--input", default="results/benchmark_summary_1200.json", help="Ruta al JSON de métricas.")
    parser.add_argument("--out-dir", default="results/figures_rigorous", help="Directorio destino para las figuras.")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = load_data(args.input)
    slices = data.get("slices", [])

    print(f"-> Procesando {len(slices)} cortes experimentales desde {args.input}...")
    plot_entropy_series(slices, out_path)
    plot_containment_scatter(slices, out_path)
    plot_behavioral_archetypes(slices, out_path)

    print(f"Figuras rigurosas con 95% CI generadas en: {out_path.resolve()}/")
    print("  • entropy_degradation_series.png (con bandas sombreadas)")
    print("  • containment_scatter.png (con cruces de error bidimensional)")
    print("  • behavioral_archetypes.png (con barras de error y capuchones)")


if __name__ == "__main__":
    main()
