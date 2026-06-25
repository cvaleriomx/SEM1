"""Simulación de transporte de electrones con solenoides y aperturas.

Orden obligatorio de inicialización de Warp:

1. Leer los argumentos antes de importar Warp.
2. Definir la especie y sus propiedades.
3. Añadir los campos magnéticos.
4. Crear los conductores (sin instalarlos todavía).
5. Configurar la malla y el paquete ``wxy``.
6. Ejecutar ``wp.generate()``.
7. Instalar los conductores y ejecutar ``wp.fieldsolve()``.
8. Avanzar las partículas y guardar los diagnósticos.
"""

import sys
import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from libreria import (
    calc_emit_rms,
    cargar_campo_poisson_3d_para_warp,
    multiplot2,
)


@dataclass(frozen=True)
class SimulationConfig:
    """Parámetros físicos y numéricos de la simulación."""

    solenoid_fields: tuple[float, float, float]
    run_length: float = 0.39
    number_of_particles: int = 1000000
    grid_cells: int = 150
    grid_radius: float = 0.011
    pipe_radius: float = 0.01
    step_size: float = 0.250e-3
    plot_results: bool = True
    use_realistic_solenoid_field: bool = False
    diagnostics_every: int = 5
    random_seed: int | None = None
    output_file: Path = Path("output.txt")
    arrays_file: Path = Path("arreglos.txt")
    profiles_directory: Path = Path("salida")
    optimization_mode: bool = False
    enable_stigmator: bool = True
    space_charge: bool = False


@dataclass
class BeamParameters:
    kinetic_energy: float
    emittance: float
    current: float
    radius_x: float
    radius_y: float
    divergence_x: float
    divergence_y: float
    velocity: float


@dataclass
class InitialDistribution:
    x: np.ndarray
    y: np.ndarray


@dataclass
class Diagnostics:
    z_mm: list[float] = field(default_factory=list)
    emittance_um: list[float] = field(default_factory=list)
    x_rms: list[float] = field(default_factory=list)
    beta_x: list[float] = field(default_factory=list)
    chi: list[float] = field(default_factory=list)
    particle_x: list[float] = field(default_factory=list)
    particle_xp: list[float] = field(default_factory=list)
    particle_y: list[float] = field(default_factory=list)
    particle_vx: list[float] = field(default_factory=list)
    particle_vy: list[float] = field(default_factory=list)
    particle_bz: list[float] = field(default_factory=list)
    particle_count: list[int] = field(default_factory=list)
    mean_rho: list[float] = field(default_factory=list)


def read_solenoid_fields(argv: list[str]) -> tuple[float, float, float]:
    """Lee los tres campos de solenoide y deja libres los argumentos de Warp."""

    if len(argv) != 3:
        program = Path(sys.argv[0]).name
        raise SystemExit(
            f"Uso: python {program} CAMPO_1 CAMPO_2 CAMPO_3\n"
            f"Ejemplo: python {program} 0.07 0.0511 0.08"
        )

    try:
        fields = tuple(float(value) for value in argv)
    except ValueError as exc:
        raise SystemExit("Los tres campos magnéticos deben ser números.") from exc

    # Warp también procesa sys.argv durante su importación. Se eliminan aquí
    # nuestros argumentos para evitar que Warp los interprete como propios.
    del sys.argv[1:]
    return fields


def create_beam_parameters(wp) -> BeamParameters:
    kinetic_energy = 30.0 * wp.keV
    emittance = 4.00e-6
    current = 0.000050 * wp.mA
    radius_x = 0.30 * wp.mm
    radius_y = 0.30 * wp.mm
    alpha_x = -7.5
    r_xp=-alpha_x*emittance/radius_x

    divergence_x = r_xp

    electron_rest_energy = 0.911e9
    speed_of_light = 299_792_458.0
    gamma = 1.0 + kinetic_energy / electron_rest_energy
    beta = np.sqrt(1.0 - 1.0 / gamma**2)

    return BeamParameters(
        kinetic_energy=kinetic_energy,
        emittance=emittance,
        current=current,
        radius_x=radius_x,
        radius_y=radius_y,
        divergence_x=divergence_x,
        divergence_y=divergence_x,
        velocity=beta * speed_of_light,
    )


def create_electron_species(wp, beam: BeamParameters):
    """Crea la especie y asigna las condiciones iniciales del haz."""

    electrons = wp.Species(
        type=wp.Electron,
        charge_state=-1,
        name="Electrons",
    )

    electrons.ekin = beam.kinetic_energy
    electrons.vbeam = 0.0
    electrons.ibeam = beam.current
    electrons.emitx = beam.emittance
    electrons.emity = beam.emittance
    electrons.vthz = 0.0
    electrons.x0 = 0.0
    electrons.y0 = 0.0
    electrons.xp0 = 0.0
    electrons.yp0 = 0.0
    electrons.a0 = beam.radius_x
    electrons.b0 = beam.radius_y
    electrons.ap0 =beam.divergence_x
    electrons.bp0 = beam.divergence_y

    wp.derivqty()
    return electrons


def create_initial_distribution(
    config: SimulationConfig,
    beam: BeamParameters,
) -> InitialDistribution:
    """Genera y ordena radialmente la distribución transversal inicial."""

    if config.random_seed is not None:
        np.random.seed(config.random_seed)

    count = config.number_of_particles

    x = np.random.normal(0.0, beam.radius_x, count)
    y = np.random.normal(0.0, beam.radius_y, count)
    radial_order = np.argsort(np.hypot(x, y))

    return InitialDistribution(
        x=x[radial_order],
        y=y[radial_order],
    )


def add_background_field(wp, field_data, scale: float, z_offset: float) -> None:
    bx, by, bz, parameters = field_data
    wp.addnewbgrd(
        parameters["z_start"] + z_offset,
        parameters["z_stop"] + z_offset,
        xs=parameters["xs"],
        dx=parameters["dx"],
        ys=parameters["ys"],
        dy=parameters["dy"],
        nx=parameters["nx"],
        ny=parameters["ny"],
        nz=parameters["nz"],
        bx=bx * scale,
        by=by * scale,
        bz=bz * scale,
    )


def configure_realistic_solenoids(wp, fields: tuple[float, float, float]) -> None:
    """Carga los mapas 3D de campo magnético calculados con Poisson."""

    transport_field = cargar_campo_poisson_3d_para_warp(
        "solenois_sem_normalizedr0.TXT",
        Nx=101,
        Ny=101,
        Nz=201,
    )
    add_background_field(wp, transport_field, fields[0], z_offset=0.0)
    add_background_field(wp, transport_field, fields[1], z_offset=0.08)

    target_field = cargar_campo_poisson_3d_para_warp(
        "objetivo_normalized_r0.TXT",
        Nx=51,
        Ny=51,
        Nz=201,
    )
    add_background_field(wp, target_field, fields[2], z_offset=0.26)


def configure_hard_edge_solenoids(wp, fields: tuple[float, float, float]) -> None:
    """Añade los tres solenoides ideales usados por la simulación original."""

    wp.addnewsolenoid(
        zi=0.08 - 25 * wp.mm / 2,
        zf=0.08 + 25 * wp.mm / 2,
        ri=0.020,
        maxbz=fields[0],
    )
    wp.addnewsolenoid(
        zi=0.16 - 25 * wp.mm / 2,
        zf=0.16 + 25 * wp.mm / 2,
        ri=0.020,
        maxbz=fields[1],
    )
    wp.addnewsolenoid(
        zi=0.30 - 8 * wp.mm / 2,
        zf=0.30 + 8 * wp.mm / 2,
        ri=0.010,
        maxbz=fields[2],
    )
    
def configure_magnetic_stigmator(
    wp,
    normal_gradient,
    skew_gradient,
    z_center=0.280,
    length=0.010,
    aperture=0.002,
):
    z_start = z_center - length / 2
    z_end = z_center + length / 2

    normal_index = wp.addnewquad(
        zs=z_start,
        ze=z_end,
        db=normal_gradient,
        ph=0.0,
        ap=aperture,
    )

    skew_index = wp.addnewquad(
        zs=z_start,
        ze=z_end,
        db=skew_gradient,
        ph=np.pi / 4,
        ap=aperture,
    )

    return normal_index, skew_index
def astigmatism_metric(electrons):
    x = electrons.getx()
    y = electrons.gety()

    sigma_xx = np.var(x)
    sigma_yy = np.var(y)
    sigma_xy = np.mean((x - np.mean(x)) * (y - np.mean(y)))

    astigmatism = np.sqrt(
        (sigma_xx - sigma_yy)**2 + 4 * sigma_xy**2
    )

    spot_size = sigma_xx + sigma_yy
    return astigmatism / spot_size

def configure_magnetic_fields(wp, config: SimulationConfig) -> None:
    if config.use_realistic_solenoid_field:
        configure_realistic_solenoids(wp, config.solenoid_fields)
    else:
        configure_hard_edge_solenoids(wp, config.solenoid_fields)


def create_conductors(wp, config: SimulationConfig):
    """Construye las aperturas y la tubería, pero aún no las instala."""

    plate_thickness = 0.003

    aperture_1 = wp.ZRoundedCylinderOut(
        radius=0.0005,
        length=plate_thickness,
        radius2=plate_thickness / 2,
        voltage=0.0,
        zcent=0.12,
    )
    aperture_2 = wp.ZCylinderOut(
        radius=0.001,
        zlower=0.25 - plate_thickness / 2,
        zupper=0.25 + plate_thickness / 2,
        voltage=0.0,
    )
    aperture_3 = wp.ZCylinderOut(
        radius=0.001,
        zlower=0.30 - plate_thickness / 2,
        zupper=0.30 + plate_thickness / 2,
        voltage=0.0,
    )
    pipe = wp.ZCylinderOut(
        radius=config.pipe_radius,
        zlower=0.0,
        zupper=config.run_length - 0.005,
        voltage=0.0,
    )

    return aperture_1 + aperture_2 + aperture_3 + pipe


def configure_warp_grid(wp, config: SimulationConfig) -> None:
    """Configura malla, partículas, integrador y solver antes de generate()."""

    wp.top.lprntpara = False
    wp.top.lpsplots = False
    wp.top.npmax = config.number_of_particles
    wp.top.prwall = config.grid_radius
    wp.top.lrelativ = False
    wp.top.relativity = 0
    wp.top.ibpush = 2
    wp.top.zbeam = 0.0

    wp.w3d.nx = config.grid_cells
    wp.w3d.ny = config.grid_cells
    wp.w3d.xmmin = -config.grid_radius
    wp.w3d.xmmax = config.grid_radius
    wp.w3d.ymmin = -config.grid_radius
    wp.w3d.ymmax = config.grid_radius
    wp.w3d.distrbtn = "KV"
    wp.w3d.xrandom = "digitrev"
    wp.w3d.vtrandom = "digitrev"
    wp.w3d.vzrandom = "digitrev"
    wp.w3d.cylinder = True
    wp.w3d.boundxy = 0
    wp.w3d.solvergeom = wp.w3d.XYgeom

    wp.wxy.ds = config.step_size
    wp.wxy.lvzchang = True


def initialize_warp(
    wp,
    config: SimulationConfig,
    conductors,
    electrons,
    distribution: InitialDistribution,
) -> None:
    """Genera Warp, carga la distribución e instala los conductores."""

    wp.package("wxy")
    wp.generate()
    if config.space_charge:
        wp.installconductors(conductors, dfill=wp.largepos)
        wp.fieldsolve()
    else:
        wp.top.fstype = -1
        wp.top.efetch[:] = 0
        wp.top.pgroup.ldodepos[:] = False


    loaded_particles = len(electrons.getx())
    if loaded_particles != len(distribution.x):
        raise RuntimeError(
            "Warp cargó un número de partículas diferente al solicitado: "
            f"{loaded_particles} != {len(distribution.x)}"
        )

    electrons.getx()[:] = distribution.x
    electrons.gety()[:] = distribution.y
    # Igual que en el programa original, Warp conserva las velocidades que
    # cargó a partir de las propiedades de la especie.

    # Este orden es crítico: installconductors necesita que generate() haya
    # inicializado previamente la malla y su descomposición.
    wp.installconductors(conductors, dfill=wp.largepos)
    scraper = wp.ParticleScraper(conductors)

    wp.fieldsolve()


def calculate_density_metrics(wp, config: SimulationConfig) -> tuple[float, float]:
    rho = wp.getrho(iz=0, solver=wp.getregisteredsolver())
    mean_rho = np.sum(rho) / (config.grid_cells + 1) ** 2

    coordinates = np.linspace(
        -config.grid_radius,
        config.grid_radius,
        config.grid_cells + 1,
    )
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    radius_squared = x_grid**2 + y_grid**2
    total_rho = np.sum(rho)

    if total_rho == 0.0:
        return mean_rho, np.nan

    mean_radius_squared = np.sum(rho * radius_squared) / total_rho
    chi = 6.0 * mean_radius_squared / 0.00182
    return mean_rho, chi


def record_diagnostics(
    wp,
    config: SimulationConfig,
    electrons,
    step_index: int,
    diagnostics: Diagnostics,
) -> None:
    x = electrons.getx()
    xp = electrons.getxp()

    if len(x) == 0:
        return

    x_rms = np.std(x)
    emittance = calc_emit_rms(x, xp)
    mean_rho, chi = calculate_density_metrics(wp, config)

    diagnostics.z_mm.append(step_index * config.step_size * 1e3)
    diagnostics.emittance_um.append(1e6 * emittance)
    diagnostics.x_rms.append(x_rms)
    diagnostics.beta_x.append(x_rms**2 / emittance if emittance != 0 else np.nan)
    diagnostics.chi.append(chi)
    diagnostics.particle_count.append(len(x))
    diagnostics.mean_rho.append(mean_rho)

    # Se conserva el seguimiento de la partícula 10 del código original.
    particle_index = min(10, len(x) - 1)
    diagnostics.particle_x.append(x[particle_index])
    diagnostics.particle_xp.append(xp[particle_index])
    diagnostics.particle_y.append(electrons.gety()[particle_index])
    diagnostics.particle_vx.append(electrons.getvx()[particle_index])
    diagnostics.particle_vy.append(electrons.getvy()[particle_index])
    diagnostics.particle_bz.append(electrons.getbz()[particle_index])


def run_simulation(wp, config: SimulationConfig, electrons) -> Diagnostics:
    diagnostics = Diagnostics()
    beam_species = [electrons]
    number_of_steps = int(np.ceil(config.run_length / config.step_size))

    if config.plot_results:
        config.profiles_directory.mkdir(parents=True, exist_ok=True)

    for step_index in range(number_of_steps):
        wp.step()
        if not config.optimization_mode:
            print(
                f"Paso={step_index} "
                f"{100 * step_index // number_of_steps}% "
                f"N={len(electrons.getx())}"
            )

        if not config.optimization_mode and step_index % config.diagnostics_every == 0:
            record_diagnostics(wp, config, electrons, step_index, diagnostics)

            if config.plot_results:
                distance_mm = step_index * config.step_size * 1e3
                profile = (
                    config.profiles_directory
                    / f"profiles_{distance_mm:08.3f}_mm.png"
                )
                multiplot2(beam_species, str(profile))

    return diagnostics


def final_beam_values(electrons) -> tuple[float, float]:
    if len(electrons.getx()) == 0:
        return np.nan, np.nan

    size_x = np.std(electrons.getx())
    emittance = calc_emit_rms(electrons.getx(), electrons.getxp())
    print("salida", size_x, emittance)
    return size_x, emittance


def final_spot_rms(electrons) -> float:
    """Radio RMS del spot final: sqrt(<x² + y²>)."""

    if len(electrons.getx()) == 0:
        return np.inf

    x = electrons.getx()
    y = electrons.gety()
    return float(np.sqrt(np.mean(x**2 + y**2)))


def save_results(
    config: SimulationConfig,
    diagnostics: Diagnostics,
    size_x: float,
    emittance: float,
) -> None:
    with config.output_file.open("a", encoding="utf-8") as output:
        output.write(
            f"{config.solenoid_fields[0]:.4f}\t"
            f"{size_x:.3e}\t{emittance:.3e}\n"
        )

    rows = zip(
        diagnostics.z_mm,
        diagnostics.emittance_um,
        diagnostics.x_rms,
        diagnostics.beta_x,
        diagnostics.chi,
        diagnostics.particle_x,
        diagnostics.particle_vx,
        diagnostics.particle_y,
        diagnostics.particle_vy,
        diagnostics.particle_bz,
        diagnostics.particle_count,
    )

    with config.arrays_file.open("w", encoding="utf-8") as arrays:
        for row in rows:
            arrays.write("\t".join(str(value) for value in row) + "\n")


def plot_diagnostics(wp, electrons, diagnostics: Diagnostics) -> None:
    figure, axes = plt.subplots(3, 1)

    axes[0].scatter(diagnostics.z_mm, diagnostics.particle_bz)
    axes[0].set_xlabel("z (mm)")
    axes[0].set_ylabel("Bz (T)")
    axes[0].set_title("Campo magnético de la partícula de prueba")

    axes[1].plot(
        diagnostics.z_mm,
        diagnostics.x_rms,
        label=f"{electrons.name}, X",
    )
    axes[1].set_xlabel("z (mm)")
    axes[1].set_ylabel("x RMS (m)")
    axes[1].set_title("Envolvente RMS")
    axes[1].legend()

    axes[2].plot(
        diagnostics.z_mm,
        diagnostics.particle_count,
        label="Partículas",
    )
    axes[2].set_xlabel("z (mm)")
    axes[2].set_ylabel("Número de partículas")
    axes[2].legend()

    figure.tight_layout()

    plt.figure()
    plt.scatter(
        electrons.getx(),
        electrons.getxp(),
        s=2,
        label=f"{electrons.name}, X",
    )
    plt.xlabel("x (m)")
    plt.ylabel("xp (rad)")
    plt.legend()

    plt.figure()
    plt.scatter(diagnostics.z_mm, diagnostics.beta_x, s=8)
    plt.xlabel("z (mm)")
    plt.ylabel("beta x")

    plt.figure()
    plt.scatter(diagnostics.particle_x, diagnostics.particle_xp, s=8)
    plt.xlabel("x de partícula de prueba (m)")
    plt.ylabel("xp de partícula de prueba (rad)")

    valid_beta = np.asarray(diagnostics.beta_x)
    valid_beta = valid_beta[np.isfinite(valid_beta) & (valid_beta != 0.0)]
    if len(valid_beta):
        phase_advance = np.sum(20 * wp.wxy.ds / (valid_beta * 2 * np.pi))
        print("phase advance is", phase_advance)

    plt.show()


def main() -> None:
    fields = read_solenoid_fields(sys.argv[1:])

    # Debe importarse después de retirar los argumentos propios del programa.
    import warp as wp

    optimization_mode = os.environ.get("SEM_OPTIMIZATION") == "1"
    random_seed_text = os.environ.get("SEM_RANDOM_SEED")
    random_seed = int(random_seed_text) if random_seed_text else None
    particles_text = os.environ.get("SEM_PARTICLES")
    number_of_particles = int(particles_text) if particles_text else 1000000

    config = SimulationConfig(
        solenoid_fields=fields,
        number_of_particles=number_of_particles,
        plot_results=not optimization_mode,
        random_seed=random_seed,
        optimization_mode=optimization_mode,
        enable_stigmator=not optimization_mode,
    )
    if not optimization_mode:
        print("Campos de los solenoides:", config.solenoid_fields)

    beam = create_beam_parameters(wp)
    electrons = create_electron_species(wp, beam)
    distribution = create_initial_distribution(config, beam)

    if not optimization_mode:
        print("Velocidad axial:", beam.velocity)
        print("Sigma inicial x/y:", np.std(distribution.x), np.std(distribution.y))

    configure_magnetic_fields(wp, config)
    if config.enable_stigmator:
        configure_magnetic_stigmator(
            wp,
            normal_gradient=0.10,  # T/m
            skew_gradient=0.0,    # T/m
        )
    conductors = create_conductors(wp, config)
    configure_warp_grid(wp, config)
    initialize_warp(wp, config, conductors, electrons, distribution)

    diagnostics = run_simulation(wp, config, electrons)
    if optimization_mode:
        print(f"SEM_OBJECTIVE={final_spot_rms(electrons):.16e}")
        print(f"SEM_PARTICLES={len(electrons.getx())}")
        return

    size_x, emittance = final_beam_values(electrons)
    save_results(config, diagnostics, size_x, emittance)

    if config.plot_results:
        plot_diagnostics(wp, electrons, diagnostics)


if __name__ == "__main__":
    main()
