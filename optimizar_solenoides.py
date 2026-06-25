"""Optimiza los tres campos de solenoide para minimizar el spot final.

Cada evaluación se ejecuta en un proceso nuevo para evitar que el estado global
de Warp de una lattice contamine la siguiente evaluación.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


OBJECTIVE_PATTERN = re.compile(r"SEM_OBJECTIVE=([0-9.eE+-]+)")
PARTICLES_PATTERN = re.compile(r"SEM_PARTICLES=(\d+)")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimiza el radio RMS final variando los tres solenoides."
    )
    parser.add_argument(
        "initial",
        nargs=3,
        type=float,
        metavar=("B1", "B2", "B3"),
        help="Campos iniciales de los tres solenoides, en teslas.",
    )
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        default=(0.0, 0.20, 0.0, 0.20, 0.0, 0.20),
        metavar=("B1_MIN", "B1_MAX", "B2_MIN", "B2_MAX", "B3_MIN", "B3_MAX"),
        help="Límites inferior y superior de cada campo, en teslas.",
    )
    parser.add_argument(
        "--min-transmission",
        type=float,
        default=0.02,
        help="Fracción mínima de partículas que debe llegar al final (default: 0.90).",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=100000,
        help="Número inicial de partículas usado para calcular la transmisión.",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=80,
        help="Número máximo aproximado de simulaciones (default: 80).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Semilla fija usada en todas las evaluaciones.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("optimizacion_solenoides.csv"),
        help="Archivo CSV donde se guardará cada evaluación.",
    )
    parser.add_argument(
        "--simulation",
        type=Path,
        default=Path(__file__).with_name("corrida_nobeam_aperture2_ordenado.py"),
        help="Ruta del programa de simulación.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Tiempo máximo en segundos para cada simulación.",
    )
    return parser.parse_args()


def make_bounds(values: list[float]) -> list[tuple[float, float]]:
    bounds = [(values[index], values[index + 1]) for index in range(0, 6, 2)]
    if any(lower >= upper for lower, upper in bounds):
        raise SystemExit("Cada límite inferior debe ser menor que el superior.")
    return bounds


class SolenoidObjective:
    def __init__(self, arguments: argparse.Namespace):
        self.arguments = arguments
        self.evaluations = 0
        self.cache: dict[tuple[float, float, float], float] = {}
        self.best_objective = np.inf
        self.best_fields: tuple[float, float, float] | None = None

        arguments.log.parent.mkdir(parents=True, exist_ok=True)
        with arguments.log.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(
                [
                    "evaluation",
                    "B1_T",
                    "B2_T",
                    "B3_T",
                    "spot_rms_m",
                    "particles",
                    "transmission",
                    "penalized_objective",
                    "status",
                ]
            )

    def __call__(self, fields_array: np.ndarray) -> float:
        fields = tuple(float(value) for value in fields_array)
        cache_key = tuple(round(value, 12) for value in fields)
        if cache_key in self.cache:
            return self.cache[cache_key]

        self.evaluations += 1
        command = [
            sys.executable,
            str(self.arguments.simulation.resolve()),
            *(f"{value:.16g}" for value in fields),
        ]
        environment = os.environ.copy()
        environment["SEM_OPTIMIZATION"] = "1"
        environment["SEM_RANDOM_SEED"] = str(self.arguments.seed)
        environment["SEM_PARTICLES"] = str(self.arguments.particles)

        status = "ok"
        spot_rms = np.inf
        particles = 0

        try:
            completed = subprocess.run(
                command,
                cwd=self.arguments.simulation.resolve().parent,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.arguments.timeout,
                check=False,
            )
            combined_output = completed.stdout + "\n" + completed.stderr
            objective_match = OBJECTIVE_PATTERN.search(combined_output)
            particles_match = PARTICLES_PATTERN.search(combined_output)

            if completed.returncode != 0:
                status = f"error_{completed.returncode}"
            elif objective_match is None or particles_match is None:
                status = "output_not_found"
            else:
                spot_rms = float(objective_match.group(1))
                particles = int(particles_match.group(1))
        except subprocess.TimeoutExpired:
            status = "timeout"

        transmission = particles / self.arguments.particles
        if not np.isfinite(spot_rms):
            penalized = 1.0
        elif transmission < self.arguments.min_transmission:
            deficit = self.arguments.min_transmission - transmission
            penalized = spot_rms + deficit**2
            status = "low_transmission"
        else:
            penalized = spot_rms

        self.cache[cache_key] = penalized
        if penalized < self.best_objective:
            self.best_objective = penalized
            self.best_fields = fields

        with self.arguments.log.open("a", newline="", encoding="utf-8") as output:
            csv.writer(output).writerow(
                [
                    self.evaluations,
                    *fields,
                    spot_rms,
                    particles,
                    transmission,
                    penalized,
                    status,
                ]
            )

        best_text = (
            "ninguno"
            if self.best_fields is None
            else ", ".join(f"{value:.8g}" for value in self.best_fields)
        )
        print(
            f"[{self.evaluations:03d}] "
            f"B=({fields[0]:.8g}, {fields[1]:.8g}, {fields[2]:.8g}) T  "
            f"spot={spot_rms * 1e6:.6g} um  "
            f"trans={100 * transmission:.2f}%  "
            f"mejor=({best_text})"
        )
        return penalized


def main() -> None:
    arguments = parse_arguments()
    bounds = make_bounds(arguments.bounds)
    initial = np.asarray(arguments.initial, dtype=float)

    for index, (value, (lower, upper)) in enumerate(zip(initial, bounds), start=1):
        if not lower <= value <= upper:
            raise SystemExit(
                f"El valor inicial B{index}={value} está fuera de "
                f"[{lower}, {upper}]."
            )

    objective = SolenoidObjective(arguments)
    result = minimize(
        objective,
        initial,
        method="Powell",
        bounds=bounds,
        options={
            "maxfev": arguments.max_evaluations,
            "xtol": 1e-5,
            "ftol": 1e-5,
            "disp": True,
        },
    )

    best_fields = objective.best_fields or tuple(float(value) for value in result.x)
    print("\nResultado")
    print(f"B1 = {best_fields[0]:.10g} T")
    print(f"B2 = {best_fields[1]:.10g} T")
    print(f"B3 = {best_fields[2]:.10g} T")
    print(f"Objetivo penalizado = {objective.best_objective:.10e}")
    print(f"Evaluaciones = {objective.evaluations}")
    print(f"Historial = {arguments.log.resolve()}")


if __name__ == "__main__":
    main()
