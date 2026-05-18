from dataclasses import dataclass, field
@dataclass
class RadioConfig:
    Bc_hz: float = 200e3
    M: int = 1
    sigma_w2_given: float | None = None
    pt_W: float = 2e-7  # per-client average TX power

    @property
    def Tc_s(self) -> float: return 1.0 / self.Bc_hz
    @property
    def Tcoord_s(self) -> float: return self.M * self.Tc_s


# NEW: CSIT-related configuration
@dataclass
class CSITConfig:
    # If you ever want a global switch
    enabled: bool = False

    # Selection rule: "snr" | "hnorm" | "beta"
    thresh_by: str = "snr"

    # Threshold on the chosen metric; None = “no drop, just keep_min”
    thresh: float | None = None

    # Minimum number of clients to keep per round
    keep_min: int = 1

    # Whether to normalize kept users to equal effective gain
    equal_gain: bool = False

    # How to normalize coefficients across kept users:
    # "none" | "meancoeff" | "sumcoeff"
    norm_mode: str = "meancoeff"


@dataclass
class ClipConfig:
    mode: str = "per_coord"  # "none" | "per_coord" | "l2"
    B: float = 0.5
    L2_max: float = 1.0

@dataclass
class GainNormConfig:
    mode: str = "pilotless"  # "none" | "pilot" | "pilotless" | "given" | "pathloss"
    e0_pilot: float = 1.0
    G_given: float | None = None
    use_expected_M_factor: bool = True

@dataclass
class NoiseConfig:
    mode: str = "awgn"   # "none" | "awgn" | "given"
    sigma_w2: float = 5e-12

@dataclass
class AlgoConfig:
    agg_mode: str = "fedavg"    # ["fedavg", "sgd"]

    sgd_eta0: float = 0.1
    sgd_t0: float = 10.0
    sgd_exp: float = 0.60
    clipping: ClipConfig = field(default_factory=ClipConfig)
    gain_norm: GainNormConfig = field(default_factory=GainNormConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    # NEW: CSIT selection config
    csit: CSITConfig = field(default_factory=CSITConfig)

@dataclass
class SimConfig:
    rounds: int = 3
    local_epochs: int = 1
    clients: int = 10
    sample_m: int = 5

@dataclass
class Config:
    radio: RadioConfig = field(default_factory=RadioConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    sim: SimConfig = field(default_factory=SimConfig)
